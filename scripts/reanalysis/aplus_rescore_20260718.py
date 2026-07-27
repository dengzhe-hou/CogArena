#!/usr/bin/env python3
"""A+ estimand rescoring and inference suite, v2.4 (2026-07-19).

v2.4 freezes the deterministic strict-v4 parser as the reported primary
operation-span estimator and the canonical whitespace parser as a scoring-
specification sensitivity.  This is a post-hoc scorer correction: the parser
rules were refined after inspecting production response formats, then frozen
before the final paper-wide recomputation.  No human annotation contributes to
the reported scores.  The blinded-adjudication machinery introduced in v2.3 is
retained only as an unused contingency and is activated exclusively by
COGARENA_PRIMARY_CONFIG=aplus_adjudicated (or the legacy
COGARENA_REQUIRE_ADJ=1 switch).

v2.3 adds the blind-adjudication protocol after the third re-verification
round found further production misparses in parser v4 (interim fragments,
explanation lines, refusal prose, and code all reachable by the
last-anchored-line rule).  This protocol was prepared but not executed for the
reported v2.4 analysis.
  - classify_response() splits the 2,750 recall responses into zero-prose
    machine classes (empty / single pure sequence / one-letter-per-line)
    and a blind class for human adjudication; the split is single-sourced
    here and consumed by scripts/reanalysis/build_ospan_blind_package.py.
  - When the contingency is explicitly activated and
    results/reanalysis/aplus_20260718/adjudications.json exists, the suite
    scores blind episodes from the frozen adjudications
    (sequence -> tokens, ambiguous/refusal/echo/non-answer -> 0) and runs
    the aplus_adjudicated primary configuration through the FULL chain:
    permutation inference, all four bootstrap layers, construct-native,
    scaling (both pools), the math-gate sensitivity, matrix CSVs, and an
    ambiguous-excluded sensitivity with its own two-level bootstrap.
    Hard validations: adjudication keys must equal the blind-mapping keys
    exactly; sequence tokens must be 1-40 single letters; every blind
    episode's response text is re-hashed against the mapping (drift
    detection); the mapping and annotator-bundle package hashes must match
    PACKAGE_MANIFEST.json.  COGARENA_REQUIRE_ADJ=1 makes a missing or
    invalid adjudication file a hard failure (set it for the formal
    freeze).  Until the file exists the hook is inert and v2.2 outputs
    are reproduced unchanged.

v2.2 (parser v4) corrects the second re-verification round against v2.1
(commit 8ebfef8):
  - Candidate ranking replaces first-run selection: the LAST anchored
    recall/answer line outranks the first answer-only line, which outranks
    verbose lines, so an interim transcript fragment can no longer defeat an
    explicit final recall line.
  - Conjunctions (and/then/&) bridge letter runs inside anchored or
    answer-only lines, so "N, K, and G" parses fully; quoted sequences are
    unwrapped.
  - The chosen candidate line is persisted by sha256 prefix alongside its
    index and tokens; four production regression fixtures join the tests.
  - The scaling section reports the 20- and 55-model pools separately with
    r/p per pool; the family-aware mixed-model refresh stays a downstream
    artifact update.
  - Dependency helpers derive ROOT from their own location when
    COGARENA_ROOT is unset (no /home path defaults).

v2.1 corrected the first re-verification round against v2 (commit 411ef15):
  (1) OSpan strict parser grammar v3. The v2 grammar rejected clear verbose
      answers ("The letters are N, K, G.", bracketed sequences, sequences
      followed by parentheticals) while accepting arbitrary colon suffixes
      ("Item 2: D"). v3 freezes an explicit answer grammar: per line, strip
      an enumerator prefix, surrounding brackets, and trailing
      parentheticals; the answer is the first maximal run of two or more
      single-letter tokens; a line whose whole residue is one letter counts
      as a single-letter answer, and consecutive such lines concatenate
      (one-letter-per-line format); a residue that is one all-caps token of
      2-8 letters (YES/NO excluded) is a compact sequence. The colon rule is
      removed. Tokens, source line, and parse status persist per episode.
      Unit tests: tests/test_aplus_ospan_parser.py.
  (2) Public gold consumed by default: the committed anonymized manifest
      (ospan_gold_manifest.json) is the default math/letter gold source; the
      private snapshot (COGARENA_SNAPSHOT) is only an optional equality
      check, so the tracked package reproduces math metrics without it.
  (3) Two-level resampling follows the authoritative order: items are
      resampled once per ORIGINAL model over all 55 rows, and family
      duplication selects rows from that matrix afterwards.
  (4) Tree hashes use ROOT-relative paths and now cover every consumed
      tree: multi-turn episodes, static details, go/no-go rerun details,
      confidence-calibration response files, and rescore overlays.
  Also new: construct-native and 20-model scaling sections so every number
  the paper update needs regenerates from this one script.

Frozen scoring contracts (A+ directive, amended v2.4):
  OSpan primary = deterministic strict-v4 positional recall partial credit
  (aplus_strict); the canonical whitespace parser is a scoring-specification
  sensitivity and is co-reported.  The strict parser is a syntactic parser,
  not a semantic ambiguity classifier;
  responses with no admissible sequence receive 0.
  Math verification is an independent process metric (manifest gold,
  score_ospan containment semantics); 75/80/85% gates are complete-case
  sensitivity analyses only.
  CVLT: recall_t = |unique(_parse_word_list(resp_t)) & gold_t| / |gold_t|;
  score_t = 1[recall_t >= 0.5]; episode accuracy = mean over
  production-designated recall turns; gold from each turn's own stimulus
  list when present, else the episode's main list.
Inference: Monte Carlo 50,000 and exact 600,600-partition permutations;
  model, merged-family, raw-family, and merged-family-by-item two-level
  bootstraps; 20,000 replicates; seeds 42/43/44; percentile CIs.
"""
import json, glob, re, csv, os, sys, gzip, hashlib
import numpy as np

ROOT = os.environ.get("COGARENA_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MARKERS = ("REMEMBER THE LETTER", "YES OR NO", "NOW RECALL ALL")
ENUM_RE = re.compile(r'^\s*(?:\d+[\.\)]\s*|[-*]\s*)')
SEP_RE = re.compile(r'[\s,;/–—-]+')

# --------------------------------------------------------------------------- #
# OSpan parsers (pure functions; unit-tested)
# --------------------------------------------------------------------------- #
ANCHOR_RE = re.compile(r'\b(recall(?:ed)?|letters?|answer|sequence|order|remember(?:ed)?)\b',
                       re.IGNORECASE)
CONJ = ('and', 'then', '&')

def _strip_wrappers(line):
    s = ENUM_RE.sub('', line.strip())
    m = re.fullmatch(r'[\[\("\'](.*)[\]\)"\']', s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]\s*$', '', s).strip()
    return s.rstrip('.!').strip()

def _extract_run(toks, bridge):
    """Longest (first maximal) run of single-letter tokens; when bridge is
    True, conjunction tokens inside a run connect rather than break it."""
    best, cur = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            cur.append(t.upper())
        elif bridge and cur and t.lower() in CONJ:
            continue
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best

def _line_answer(line):
    """-> (letters, kind) with kind in {'run','single','compact',None} plus
    (anchored, answer_only) flags for candidate ranking."""
    core = _strip_wrappers(line)
    anchored = bool(ANCHOR_RE.search(line))
    if not core:
        return [], None, anchored, False
    toks = [t.strip('"\'`') for t in SEP_RE.split(core)]
    toks = [t for t in toks if t]
    answer_only = bool(toks) and all(
        (len(t) == 1 and t.isalpha()) or t.lower() in CONJ for t in toks)
    run = _extract_run(toks, bridge=(anchored or answer_only))
    if len(run) >= 2:
        return run, 'run', anchored, answer_only
    if len(toks) == 1 and len(toks[0]) == 1 and toks[0].isalpha():
        return [toks[0].upper()], 'single', anchored, True
    if re.fullmatch(r'[A-Z]{2,8}', core) and core not in ('YES', 'NO'):
        return list(core), 'compact', anchored, True
    return [], None, anchored, False

def strict_parse(text):
    """Frozen v4 strict parser. -> (tokens, status, line_index).

    Candidate ranking: the LAST anchored recall/answer line with a letter run
    ('final') outranks the first answer-only line, which outranks the first
    verbose line with a run; consecutive single-letter lines concatenate
    (one-letter-per-line). Marker lines (prompt/transcript echoes of
    "Remember the letter", "YES or NO", "Now recall ALL") never contribute;
    interim transcript fragments are therefore outranked by an explicit
    final recall line rather than truncating it."""
    lines = (text or '').splitlines()
    finals, answer_onlys, verboses, singles = [], [], [], []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if any(mk in line.upper() for mk in MARKERS):
            continue
        letters, kind, anchored, answer_only = _line_answer(line)
        if kind is None:
            continue
        if kind in ('run', 'compact'):
            if anchored:
                finals.append((letters, idx))
            elif answer_only:
                answer_onlys.append((letters, idx))
            else:
                verboses.append((letters, idx))
        elif kind == 'single':
            singles.append((letters, idx))
    if finals:
        letters, idx = finals[-1]
        return letters, 'final', idx
    if answer_onlys:
        letters, idx = answer_onlys[0]
        return letters, 'inline' if len(letters) > 1 else 'compact', idx
    if verboses:
        letters, idx = verboses[0]
        return letters, 'inline', idx
    if singles:
        seq, idx0 = list(singles[0][0]), singles[0][1]
        prev = idx0
        for letters, idx in singles[1:]:
            gap = [l for l in lines[prev + 1:idx] if l.strip()]
            if gap:
                break
            seq += letters
            prev = idx
        return seq, ('multiline' if len(seq) > 1 else 'single'), idx0
    return [], 'none', -1

def canonical_parse(text):
    """score_ospan semantics: strip, uppercase, whitespace split."""
    return (text or '').strip().upper().split()

def positional_credit(tokens, gold):
    return sum(1 for i, L in enumerate(gold) if i < len(tokens) and tokens[i] == L) / max(len(gold), 1)

# --------------------------------------------------------------------------- #
# Frozen machine/blind split for the adjudication protocol.  Machine classes
# cover only zero-prose answer formats, where any defensible parser agrees;
# every response with any surrounding language, conjunctions, parentheticals,
# multiple candidate sequences, refusals, echoes, or markup goes to blind
# human adjudication.

PURE_SEP_RE = re.compile(r'[,\s;>\-→.]+')

def _pure_line_letters(line):
    """Letters iff the raw line is a zero-prose sequence: optional enumerator,
    optional full bracket/quote wrap, letter tokens with pure separators,
    optional trailing period.  No conjunctions, no anchor words, no prose."""
    s = ENUM_RE.sub('', line.strip())
    m = re.fullmatch(r'[\[\("\'](.*)[\]\)"\']', s)
    if m:
        s = m.group(1).strip()
    s = s.rstrip('.!').strip()
    if not s:
        return None
    if re.fullmatch(r'[A-Z]{2,8}', s) and s not in ('YES', 'NO'):
        return list(s)
    toks = [t for t in PURE_SEP_RE.split(s) if t]
    if toks and all(len(t) == 1 and t.isalpha() for t in toks):
        return [t.upper() for t in toks]
    return None

def adjudicated_credit(label, tokens, gold):
    """Frozen adjudication scoring: sequence -> serial-position credit on the
    adjudicated tokens; every other label (ambiguous/refusal/echo/non-answer)
    -> 0 in the primary metric."""
    if label == 'sequence':
        return positional_credit([t.upper() for t in tokens], gold)
    return 0.0


def classify_response(text):
    """-> (cls, tokens): cls in {'machine_empty','machine_sequence',
    'machine_multiline','blind'}; tokens are the machine parse for machine
    classes (None for blind)."""
    lines = [l for l in (text or '').splitlines() if l.strip()]
    if not lines:
        return 'machine_empty', []
    parsed = [_pure_line_letters(l) for l in lines]
    if any(p is None for p in parsed):
        return 'blind', None
    if len(parsed) == 1:
        return 'machine_sequence', parsed[0]
    if all(len(p) == 1 for p in parsed):
        return 'machine_multiline', [p[0] for p in parsed]
    return 'blind', None

# --------------------------------------------------------------------------- #
def main():
    sys.path.insert(0, f"{ROOT}/scripts")
    sys.path.insert(0, f"{ROOT}/scripts/reanalysis")
    sys.path.insert(0, f"{ROOT}/results/twolevel_bootstrap_20260712")
    sys.path.insert(0, f"{ROOT}/results/recompute_20260703")
    sys.path.insert(0, f"{ROOT}/results/construct_native_20260711")
    sys.path.insert(0, ROOT)
    os.environ.setdefault("COGARENA_ROOT", ROOT)
    import compute_b2_expanded as b2
    import two_level_bootstrap as tl
    from build_construct_matrix import load_conf_cal_gold
    from cogarena.dimensions.episodic_memory import _parse_word_list

    OUT = f"{ROOT}/results/reanalysis/aplus_20260718"
    os.makedirs(OUT, exist_ok=True)
    SNAPSHOT = os.environ.get("COGARENA_SNAPSHOT",
        os.path.expanduser("~/Research/CogArena-private-archive/battery_snapshot_seed42_20260715.json.gz"))
    GOLD_MANIFEST = f"{OUT}/ospan_gold_manifest.json"

    DM = b2.DOMAIN_MAP
    paradigms = b2.PARADIGMS_ORDER

    def sha_file(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for ch in iter(lambda: f.read(1 << 20), b''):
                h.update(ch)
        return h.hexdigest()

    def sha_tree(files):
        h = hashlib.sha256()
        for f in sorted(files, key=lambda p: os.path.relpath(p, ROOT)):
            h.update(os.path.relpath(f, ROOT).encode()); h.update(b'\0')
            h.update(sha_file(f).encode()); h.update(b'\0')
        return h.hexdigest()

    rows = list(csv.reader(open(f"{ROOT}/results/recompute_20260703/corrected_matrix.csv")))
    assert rows[0][1:] == paradigms
    models = [r[0] for r in rows[1:]]
    M = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    OI = paradigms.index('operation_span'); CI = paradigms.index('cvlt_word_list')

    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    fam_raw = {m: (b2.OLD_MODELS[m][1] if m in b2.OLD_MODELS else new_meta[m].get('family', m)) for m in models}
    fam_merged = {m: re.sub(r'[\d.]+$', '', f) for m, f in fam_raw.items()}

    # ---- gold: public manifest first, snapshot as optional equality check --
    gold = {}
    gold_source = 'none'
    if os.path.exists(GOLD_MANIFEST):
        gold = json.load(open(GOLD_MANIFEST)); gold_source = 'public_manifest'
    if os.path.exists(SNAPSHOT):
        snap = json.load(gzip.open(SNAPSHOT))
        sg = {}
        for it in (snap if isinstance(snap, list) else snap.get('items', [])):
            if it.get('paradigm') != 'operation_span':
                continue
            turns = it['parameters']['turns']
            sg[it['task_id']] = {
                'letters': [l.upper() for l in it['parameters']['letters']],
                'math_expected': [t.get('math_expected') for t in turns
                                  if t.get('type') == 'operation_letter']}
        if gold:
            assert gold == sg, "public gold manifest deviates from private snapshot"
            gold_source = 'public_manifest+snapshot_verified'
        else:
            gold = sg; gold_source = 'snapshot_only'
    if not gold:
        print("WARNING: no OSpan gold available; math metrics skipped")

    # Reported freeze: strict v4 is primary.  The adjudication hook is retained
    # only as an explicit contingency and must never activate merely because a
    # stale adjudications.json happens to be present.
    ADJ_PATH = f"{OUT}/adjudications.json"
    BLIND_DIR = f"{ROOT}/results/reanalysis/ospan_blind_20260718"
    BLIND_MAP_PATH = f"{BLIND_DIR}/blind_mapping.json"
    REQUIRE_ADJ = os.environ.get('COGARENA_REQUIRE_ADJ') == '1'
    requested_primary = os.environ.get('COGARENA_PRIMARY_CONFIG')
    PRIMARY_CONFIG = requested_primary or ('aplus_adjudicated' if REQUIRE_ADJ
                                            else 'aplus_strict')

    def _req(cond, msg):
        """Estimand-critical gate: raises unconditionally (assert would be
        stripped by python -O)."""
        if not cond:
            raise RuntimeError(f"adjudication gate: {msg}")

    _req(PRIMARY_CONFIG in ('aplus_strict', 'aplus_adjudicated'),
         f"unknown COGARENA_PRIMARY_CONFIG={PRIMARY_CONFIG!r}")
    _req(not REQUIRE_ADJ or PRIMARY_CONFIG == 'aplus_adjudicated',
         "COGARENA_REQUIRE_ADJ=1 conflicts with a non-adjudicated primary")
    USE_ADJ = PRIMARY_CONFIG == 'aplus_adjudicated'
    if not USE_ADJ:
        stale_adj = [p for p in (
            ADJ_PATH,
            f"{OUT}/ADJUDICATION_MANIFEST.json",
            f"{OUT}/matrix_aplus_adjudicated.csv",
            f"{OUT}/matrix_aplus_adjudicated_exclamb.csv",
            f"{OUT}/matrix_construct_aplus_adjudicated.csv",
            f"{BLIND_DIR}/annotator_A.json",
            f"{BLIND_DIR}/annotator_B.json",
            f"{BLIND_DIR}/agreement_report.json",
            f"{BLIND_DIR}/disagreement_manifest.json",
        ) if os.path.exists(p)]
        _req(not stale_adj,
             "strict-primary freeze refuses stale adjudication artifacts: "
             f"{[os.path.basename(p) for p in stale_adj]}")

    adj_by_key = None
    from collections import Counter as _Counter
    adj_consumed = _Counter()
    adj_meta = None
    if USE_ADJ:
        _req(os.path.exists(ADJ_PATH),
             "COGARENA_REQUIRE_ADJ/adjudicated primary requires adjudications.json")
        adj_raw = json.load(open(ADJ_PATH))
        bmap = json.load(open(BLIND_MAP_PATH))
        extra = set(adj_raw) - set(bmap)
        missing = set(bmap) - set(adj_raw)
        _req(not extra, f"adjudications.json has unknown blind_ids: {sorted(extra)[:5]}")
        _req(not missing, f"adjudications.json missing blind_ids: {sorted(missing)[:5]}")
        pkg_manifest = json.load(open(f"{BLIND_DIR}/PACKAGE_MANIFEST.json"))
        _req(sha_file(BLIND_MAP_PATH) == pkg_manifest['blind_mapping_sha256'],
             "blind_mapping.json hash differs from PACKAGE_MANIFEST")
        pkg_path = f"{BLIND_DIR}/annotator_bundle/blind_package.jsonl"
        instr_path = f"{BLIND_DIR}/annotator_bundle/INSTRUCTIONS.md"
        _req(os.path.exists(pkg_path), "annotator_bundle/blind_package.jsonl missing")
        _req(os.path.exists(instr_path), "annotator_bundle/INSTRUCTIONS.md missing")
        _req(sha_file(pkg_path) == pkg_manifest['blind_package_sha256'],
             "blind_package.jsonl hash differs from PACKAGE_MANIFEST")
        _req(sha_file(instr_path) == pkg_manifest['instructions_sha256'],
             "INSTRUCTIONS.md hash differs from PACKAGE_MANIFEST")
        if USE_ADJ:
            am_path = f"{OUT}/ADJUDICATION_MANIFEST.json"
            _req(os.path.exists(am_path),
                 "ADJUDICATION_MANIFEST.json (dual-annotator provenance) is missing")
            am = json.load(open(am_path))
            _req(am.get('consensus_sha256') == sha_file(ADJ_PATH),
                 "adjudications.json differs from the consensus recorded in ADJUDICATION_MANIFEST")
            for k in ('annotator_A_sha256', 'annotator_B_sha256',
                      'disagreement_manifest_sha256', 'package_manifest_sha256'):
                _req(am.get(k), f"ADJUDICATION_MANIFEST missing {k}")
            ag = am.get('agreement') or {}
            for k in ('label_agreement', 'sequence_token_exact_agreement', 'n_disagreements'):
                _req(k in ag, f"ADJUDICATION_MANIFEST agreement missing {k}")
            _req(am['package_manifest_sha256'] == sha_file(f"{BLIND_DIR}/PACKAGE_MANIFEST.json"),
                 "ADJUDICATION_MANIFEST binds a different PACKAGE_MANIFEST than the one on disk")
            _req(am['blind_mapping_sha256'] == sha_file(BLIND_MAP_PATH),
                 "ADJUDICATION_MANIFEST binds a different blind_mapping than the one on disk")
            for key, fname in (('annotator_A_sha256', 'annotator_A.json'),
                               ('annotator_B_sha256', 'annotator_B.json'),
                               ('disagreement_manifest_sha256', 'disagreement_manifest.json')):
                fpath = f"{BLIND_DIR}/{fname}"
                _req(os.path.exists(fpath), f"{fname} required on disk")
                _req(sha_file(fpath) == am[key],
                     f"{fname} on disk differs from the hash in ADJUDICATION_MANIFEST")
            rep_path = f"{BLIND_DIR}/agreement_report.json"
            _req(os.path.exists(rep_path), "agreement_report.json required")
            _req(am.get('agreement_report_sha256') == sha_file(rep_path),
                 "agreement_report.json on disk differs from ADJUDICATION_MANIFEST binding")
            rep = json.load(open(rep_path))
            _req(rep['annotator_A_sha256'] == am['annotator_A_sha256']
                 and rep['annotator_B_sha256'] == am['annotator_B_sha256'],
                 "agreement_report binds different annotator files than ADJUDICATION_MANIFEST")
            _req(sha_file(f"{BLIND_DIR}/redaction_audit.json")
                 == pkg_manifest['redaction_audit_sha256'],
                 "redaction_audit.json drifted from PACKAGE_MANIFEST")

            # Re-execute the consensus invariant and the agreement numbers
            # from the on-disk A/B files: hash checks alone cannot catch a
            # consensus that silently overrides an agreed annotation.
            annA = json.load(open(f"{BLIND_DIR}/annotator_A.json"))
            annB = json.load(open(f"{BLIND_DIR}/annotator_B.json"))
            _req(set(annA) == set(bmap) and set(annB) == set(bmap),
                 "annotator files do not cover the blind_id set exactly")
            lab_agree = seq_both = tok_agree = 0
            for bid in bmap:
                a, b, c = annA[bid], annB[bid], adj_raw[bid]
                same_label = a['label'] == b['label']
                agreed = same_label and (a['label'] != 'sequence'
                                         or a.get('tokens') == b.get('tokens'))
                if agreed:
                    _req(c['label'] == a['label']
                         and (a['label'] != 'sequence'
                              or c.get('tokens') == a.get('tokens')),
                         f"consensus overrides the agreed annotation for {bid}")
                if same_label:
                    lab_agree += 1
                if a['label'] == b['label'] == 'sequence':
                    seq_both += 1
                    if a.get('tokens') == b.get('tokens'):
                        tok_agree += 1
            _req(abs(rep['label_agreement'] - lab_agree / len(bmap)) < 1e-9,
                 "agreement_report label_agreement does not match recomputation from A/B")
            tok_val = (tok_agree / seq_both) if seq_both else None
            got_tok = rep['sequence_token_exact_agreement']
            _req((tok_val is None and got_tok is None)
                 or (tok_val is not None and got_tok is not None
                     and abs(got_tok - tok_val) < 1e-9),
                 "agreement_report token agreement does not match recomputation from A/B")
        adj_by_key = {}
        lab_counts = {}
        for bid, ent in bmap.items():
            a = adj_raw[bid]
            lab = a['label']
            _req(lab in ('sequence', 'ambiguous', 'refusal', 'echo', 'non-answer'),
                 f"bad label {lab} for {bid}")
            toks = a.get('tokens')
            if lab == 'sequence':
                _req(isinstance(toks, list) and 1 <= len(toks) <= 40
                     and all(isinstance(t, str) and re.fullmatch(r'[A-Z]', t)
                             for t in toks), f"bad tokens schema for {bid}: {toks!r}")
            else:
                _req(not toks, f"tokens given for non-sequence label {bid}")
            lab_counts[lab] = lab_counts.get(lab, 0) + len(ent['episodes'])
            rec = {'label': lab, 'tokens': toks or [],
                   'text_sha256': ent['text_sha256'], 'blind_id': bid}
            for ep in ent['episodes']:
                key = (ep['model'], ep['task_id'])
                _req(key not in adj_by_key,
                     f"blind mapping collision: {key} mapped by {adj_by_key[key]['blind_id'] if key in adj_by_key else ''} and {bid}")
                adj_by_key[key] = rec
        adj_meta = {'file_sha256': sha_file(ADJ_PATH),
                    'blind_mapping_sha256': sha_file(BLIND_MAP_PATH),
                    'n_blind_ids': len(adj_raw),
                    'episode_label_counts': lab_counts}
    if USE_ADJ:
        _req(adj_by_key is not None,
             "adjudicated primary selected but adjudications.json is missing")

    def ep_paths(model, paradigm, dim):
        for base in (f"{ROOT}/results/multiturn_eval_v3/openai_{model}/{dim}/{paradigm}",
                     f"{ROOT}/results/multiturn_expansion/openai_{model}/text/{dim}/{paradigm}"):
            fs = sorted(glob.glob(base + "/*.json"))
            if fs:
                return fs
        return []

    def rescore_ospan(model):
        out = []
        for f in ep_paths(model, 'operation_span', 'working_memory'):
            d = json.load(open(f)); resp = d['responses']; tid = d['task_id']
            letters = [mm.group(1).upper() for t in resp[:-1]
                       if (mm := re.search(r'Remember the letter:\s*([A-Za-z])', t['stimulus']))]
            g = gold.get(tid)
            if g:
                assert [l.upper() for l in g['letters']] == letters, f"letter gold mismatch {model}/{tid}"
            text = resp[-1]['response'] or ''
            canon = positional_credit(canonical_parse(text), letters)
            toks, status, line_idx = strict_parse(text)
            strict = positional_credit(toks, letters)
            math_acc = None
            if g:
                mc = 0
                rs = [t['response'] or '' for t in resp[:-1]]
                for e, r in zip(g['math_expected'], rs):
                    ru = (r or '').upper()
                    if e == 'YES' and 'YES' in ru:
                        mc += 1
                    elif e == 'NO' and 'NO' in ru:
                        mc += 1
                math_acc = mc / max(len(g['math_expected']), 1)
            lines = text.splitlines()
            line_txt = lines[line_idx] if 0 <= line_idx < len(lines) else ''
            mclass, mtoks = classify_response(text)
            entry = {'task_id': tid, 'strict': strict, 'canonical': canon,
                     'math': math_acc, 'parse_status': status,
                     'parse_line': line_idx, 'parse_tokens': ''.join(toks),
                     'parse_line_sha16': hashlib.sha256(line_txt.encode()).hexdigest()[:16],
                     'machine_class': mclass}
            if adj_by_key is not None:
                # The blind mapping is the AUTHORITATIVE set: every mapped
                # episode takes its human adjudication and is hash-checked
                # against the frozen raw text, regardless of what the
                # current classifier says (a classifier drift must not
                # bypass the mapping); unmapped episodes must classify as
                # machine.
                a = adj_by_key.get((model, tid))
                if a is not None:
                    got = hashlib.sha256(text.encode()).hexdigest()
                    _req(got == a['text_sha256'],
                         f"response drift for {model}/{tid}: {got[:16]} != mapped ({a['blind_id']})")
                    adj_consumed[(model, tid)] += 1
                    entry['adj_strict'] = adjudicated_credit(a['label'], a['tokens'], letters)
                    entry['adj_label'] = a['label']
                else:
                    _req(mclass != 'blind',
                         f"blind-classified episode missing from the mapping: {model}/{tid}")
                    entry['adj_strict'] = positional_credit(mtoks, letters)
                    entry['adj_label'] = mclass
            out.append(entry)
        return out

    def rescore_cvlt(model):
        out = []; over_prod = 0
        for f in ep_paths(model, 'cvlt_word_list', 'episodic_memory'):
            d = json.load(open(f)); resp = d['responses']
            main_list = None; stim_lists = {}
            for i, t in enumerate(resp):
                mm = re.search(r'(?:list of words carefully:|NEW list of words \(different from before\):)\s*\n\s*\n?(.+)',
                               t['stimulus'])
                if mm:
                    lst = [w.strip().lower() for w in mm.group(1).split(',') if w.strip()]
                    stim_lists[i] = lst
                    if main_list is None and 'Learning Trial' in t['stimulus']:
                        main_list = lst
            designated = [ts for ts in d['score'].get('turn_scores', [])
                          if ts.get('scored') is not False and 'recall' in ts]
            assert len(designated) == d['score'].get('n_scored'), \
                f"designated recall turns != n_scored in {model}/{d['task_id']}"
            scores = []
            for ts in designated:
                if ts.get('hits', 0) > ts.get('total', 10**9):
                    over_prod += 1
                i = ts['trial'] - 1
                gd = stim_lists.get(i, main_list)
                if not gd:
                    continue
                toks = set(_parse_word_list(resp[i]['response'] or ''))
                recall = len(toks & set(gd)) / len(gd)
                scores.append(1.0 if recall >= 0.5 else 0.0)
            out.append({'task_id': d['task_id'],
                        'fixed_accuracy': float(np.mean(scores)) if scores else 0.0})
        return out, over_prod

    ospan = {}; cvlt = {}; over_production = 0
    for m in models:
        ospan[m] = rescore_ospan(m)
        cvlt[m], ov = rescore_cvlt(m); over_production += ov
    if adj_by_key is not None:
        leftover = set(adj_by_key) - set(adj_consumed)
        _req(not leftover,
             f"mapped episodes never consumed ({len(leftover)}): {sorted(leftover)[:5]}")
        multi = {k: c for k, c in adj_consumed.items() if c != 1}
        _req(not multi,
             f"mapped episodes consumed more than once: {sorted(multi.items())[:5]}")
        extra_keys = set(adj_consumed) - set(adj_by_key)
        _req(not extra_keys, f"unmapped keys consumed: {sorted(extra_keys)[:5]}")

    col_s = np.array([np.mean([e['strict'] for e in ospan[m]]) for m in models])
    col_c = np.array([np.mean([e['canonical'] for e in ospan[m]]) for m in models])
    col_cv = np.array([np.mean([e['fixed_accuracy'] for e in cvlt[m]]) for m in models])
    math_all = np.array([e['math'] for m in models for e in ospan[m] if e['math'] is not None])
    status_counts = {}
    for m in models:
        for e in ospan[m]:
            status_counts[e['parse_status']] = status_counts.get(e['parse_status'], 0) + 1

    MA = M.copy(); MA[:, OI] = col_s; MA[:, CI] = col_cv
    MC = M.copy(); MC[:, OI] = col_c; MC[:, CI] = col_cv

    def zscore(X): return (X - X.mean(0)) / X.std(0, ddof=0)
    LAB = [DM[p] for p in paradigms]
    PAIRS = [(i, j) for i in range(13) for j in range(i + 1, 13)]
    W_MASK = np.array([LAB[i] == LAB[j] for i, j in PAIRS])

    def delta_of(corr):
        v = np.array([corr[i, j] for i, j in PAIRS])
        return float(v[W_MASK].mean() - v[~W_MASK].mean())

    def perm_mc(corr, n=50000, seed=42):
        d = delta_of(corr); rng = np.random.default_rng(seed)
        v = np.array([corr[i, j] for i, j in PAIRS]); ds = []
        for _ in range(n):
            sh = rng.permutation(LAB)
            mask = np.array([sh[i] == sh[j] for i, j in PAIRS])
            ds.append(v[mask].mean() - v[~mask].mean())
        ds = np.array(ds)
        return d, float((ds >= d).mean()), float((np.abs(ds) >= abs(d)).mean())

    def perm_exact(corr):
        import itertools
        v = {(i, j): corr[i, j] for i, j in PAIRS}
        idx = list(range(13)); d_obs = delta_of(corr); S_tot = sum(v.values())
        deltas = []
        for nine in itertools.combinations(idx, 9):
            rest4 = [x for x in idx if x not in nine]
            a0 = nine[0]
            for g1r in itertools.combinations(nine[1:], 2):
                g1 = {a0, *g1r}
                rem6 = [x for x in nine if x not in g1]
                b0 = rem6[0]
                for g2r in itertools.combinations(rem6[1:], 2):
                    g2 = {b0, *g2r}
                    g3 = [x for x in rem6 if x not in g2]
                    c0 = rest4[0]
                    for dp in rest4[1:]:
                        h1 = {c0, dp}
                        h2 = [x for x in rest4 if x not in h1]
                        Sw = 0.0
                        for grp in (tuple(sorted(g1)), tuple(sorted(g2)), tuple(sorted(g3))):
                            Sw += v[(grp[0], grp[1])] + v[(grp[0], grp[2])] + v[(grp[1], grp[2])]
                        Sw += v[tuple(sorted(h1))] + v[tuple(sorted(h2))]
                        deltas.append(Sw / 11.0 - (S_tot - Sw) / 67.0)
        ds = np.array(deltas)
        assert len(ds) == 600600
        return d_obs, float((ds >= d_obs).mean()), float((np.abs(ds) >= abs(d_obs)).mean())

    def pc1_removed(X):
        Z = zscore(X); U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        return Z - np.outer(U[:, 0] * S[0], Vt[0])

    def pc1_share(X):
        ev = np.linalg.eigvalsh(np.corrcoef(zscore(X).T))[::-1]
        return float(ev[0] / ev.sum())

    cc_gold = load_conf_cal_gold()

    def build_pools(parser_key, target, exclude_ambiguous=False):
        pools = {}; bad = []
        for mi, m in enumerate(models):
            pl = tl.item_pools(m, m in b2.OLD_MODELS, cc_gold)
            assert pl is not None, f"pools missing for {m}"
            pl['operation_span'] = [(e['task_id'], e[parser_key]) for e in ospan[m]
                                    if not (exclude_ambiguous and e['adj_label'] == 'ambiguous')]
            pl['cvlt_word_list'] = [(e['task_id'], e['fixed_accuracy']) for e in cvlt[m]]
            for pi, p in enumerate(paradigms):
                assert p in pl and pl[p], f"empty pool {m}/{p}"
                mean = float(np.mean([a for _, a in pl[p]]))
                if abs(mean - target[mi, pi]) > tl.TOL:
                    bad.append((m, p, round(mean, 6), round(float(target[mi, pi]), 6)))
            pools[m] = pl
        assert not bad, f"SELF-CHECK FAILED: {bad[:8]} (+{max(0, len(bad)-8)} more)"
        crossed = {}; item_mats = {}; per_model = {}
        for p in paradigms:
            idsets = [frozenset(k for k, _ in pools[m][p]) for m in models]
            crossed[p] = all(s == idsets[0] for s in idsets)
            if crossed[p]:
                ids = sorted(idsets[0])
                item_mats[p] = np.array([[dict(pools[m][p])[k] for k in ids] for m in models])
            else:
                per_model[p] = [np.array([a for _, a in pools[m][p]]) for m in models]
        return crossed, item_mats, per_model

    def twolevel_matrix(rng, tw):
        """Authoritative order: item-resample once per ORIGINAL model row."""
        crossed, item_mats, per_model = tw
        out = np.empty((len(models), 13))
        for pi, p in enumerate(paradigms):
            if crossed[p]:
                P = item_mats[p]
                ii = rng.integers(0, P.shape[1], P.shape[1])
                out[:, pi] = P[:, ii].mean(1)
            else:
                for i in range(len(models)):
                    arr = per_model[p][i]
                    ii = rng.integers(0, arr.size, arr.size)
                    out[i, pi] = arr[ii].mean()
        return out

    def boot_ci(X, mode, tw=None, n_boot=20000, seed=42):
        rng = np.random.default_rng(seed)
        fams = fam_raw if mode == 'family_raw' else fam_merged
        fam_ids = sorted(set(fams.values()))
        idx_by = {f: [i for i, m in enumerate(models) if fams[m] == f] for f in fam_ids}
        ds = []
        for _ in range(n_boot):
            if mode == 'model':
                Xb = X[rng.choice(55, 55, replace=True)]
            elif mode in ('family_merged', 'family_raw'):
                pick = rng.choice(len(fam_ids), len(fam_ids), replace=True)
                Xb = X[[i for k in pick for i in idx_by[fam_ids[k]]]]
            else:
                mat = twolevel_matrix(rng, tw)
                pick = rng.choice(len(fam_ids), len(fam_ids), replace=True)
                Xb = mat[[i for k in pick for i in idx_by[fam_ids[k]]]]
            if np.any(Xb.std(0) == 0):
                continue
            ds.append(delta_of(np.corrcoef(Xb.T)))
        return [float(x) for x in np.percentile(ds, [2.5, 97.5])], len(ds)

    mclass_counts = {}
    for m in models:
        for e in ospan[m]:
            mclass_counts[e['machine_class']] = mclass_counts.get(e['machine_class'], 0) + 1

    results = {'spec': 'A+ 2026-07-19 v2.4 (strict-v4 primary; canonical sensitivity)',
               'n_boot': 20000, 'seeds': [42, 43, 44],
               'quantile_method': 'numpy percentile linear',
               'gold_source': gold_source,
               'ospan_parse_status_counts': status_counts,
               'ospan_machine_blind_counts': mclass_counts,
               'adjudication': adj_meta,
               'adjudication_status': ('performed' if adj_by_key is not None else 'not_performed'),
               'primary_config': PRIMARY_CONFIG,
               'sensitivity_configs': ['aplus_canonical'],
               'canonical_sensitivity_scope': ('headline exact/Monte-Carlo inference, four bootstrap '
                                               'layers, construct-native analysis, and scaling'),
               'parser_freeze_timing': ('post-hoc scorer correction after inspection of production '
                                        'response formats; frozen before final paper-wide recomputation'),
               'cvlt_production_hits_gt_total_designated': over_production}

    configs = [('base', M, None, False), ('aplus_strict', MA, 'strict', False),
               ('aplus_canonical', MC, 'canonical', False)]
    col_a = None
    if adj_by_key is not None:
        col_a = np.array([np.mean([e['adj_strict'] for e in ospan[m]]) for m in models])
        MJ = M.copy(); MJ[:, OI] = col_a; MJ[:, CI] = col_cv
        configs.append(('aplus_adjudicated', MJ, 'adj_strict', False))
        col_ax = np.array([np.mean([e['adj_strict'] for e in ospan[m]
                                    if e['adj_label'] != 'ambiguous']) for m in models])
        MJX = M.copy(); MJX[:, OI] = col_ax; MJX[:, CI] = col_cv
        configs.append(('aplus_adjudicated_exclamb', MJX, 'adj_strict', True))

    for tag, X, pkey, excl_amb in configs:
        d, p1m, p2m = perm_mc(np.corrcoef(X.T))
        _, p1e, p2e = perm_exact(np.corrcoef(X.T))
        dr, _, pr2 = perm_mc(np.corrcoef(pc1_removed(X).T))
        _, _, pr2e = perm_exact(np.corrcoef(pc1_removed(X).T))
        ent = {'delta': round(d, 4), 'p1_mc50k': round(p1m, 4), 'p2_mc50k': round(p2m, 4),
               'p1_exact': round(p1e, 5), 'p2_exact': round(p2e, 5),
               'pc1_share': round(pc1_share(X), 4), 'pc1rm_delta': round(dr, 4),
               'pc1rm_p2_exact': round(pr2e, 5),
               'ospan_mean': round(float(X[:, OI].mean()), 4),
               'cvlt_mean': round(float(X[:, CI].mean()), 4), 'boot': {}}
        tw = build_pools(pkey, X, exclude_ambiguous=excl_amb) if pkey else None
        if tw:
            ent['noncrossed_fallback'] = [p for p in paradigms if not tw[0][p]]
        for mode in ('model', 'family_merged', 'family_raw', 'twolevel'):
            if tw is None and mode == 'twolevel':
                continue
            per_seed = {}
            for seed in (42, 43, 44):
                ci_, nb = boot_ci(X, mode, tw=tw, n_boot=20000, seed=seed)
                per_seed[seed] = {'ci': [round(ci_[0], 4), round(ci_[1], 4)], 'n_eff': nb}
            ent['boot'][mode] = per_seed
        results[tag] = ent
        print(tag, json.dumps({k: ent[k] for k in ('delta', 'p2_exact', 'pc1_share', 'ospan_mean', 'cvlt_mean')}))

    if len(math_all):
        results['math_process'] = {
            'mean': round(float(math_all.mean()), 4),
            'coverage': {str(g): round(float((math_all >= g).mean()), 4) for g in (0.75, 0.80, 0.85)},
            'n_episodes': int(len(math_all)),
            'note': 'legacy substring containment semantics (score_ospan); gold from manifest math_expected'}
        def gate_block(score_key, base_mat):
            gs = {}
            for gate in (0.75, 0.80, 0.85):
                colg = []
                for m in models:
                    vals = [e[score_key] for e in ospan[m]
                            if e['math'] is not None and e['math'] >= gate]
                    colg.append(float(np.mean(vals)) if vals else None)
                keep = [i for i, x in enumerate(colg) if x is not None]
                Xg = base_mat[keep].copy()
                Xg[:, OI] = np.array([colg[i] for i in keep])
                d, p1, p2 = perm_mc(np.corrcoef(Xg.T))
                gs[str(gate)] = {'complete_case_n': len(keep),
                                 'delta': round(d, 4), 'p2_mc': round(p2, 4)}
            return gs

        results['math_gate_sensitivity_complete_case'] = gate_block('strict', MA)
        if adj_by_key is not None:
            results['math_gate_sensitivity_complete_case_adjudicated'] = gate_block('adj_strict', MJ)
    else:
        results['math_process'] = None
        results['math_gate_sensitivity_complete_case'] = None

    # ---- construct-native and scaling sections ---------------------------- #
    crows = list(csv.reader(open(f"{ROOT}/results/construct_native_20260711/construct_matrix.csv")))
    cpar = crows[0][1:]; cmodels = [r[0] for r in crows[1:]]
    C = np.array([[float(v) for v in r[1:]] for r in crows[1:]])[[cmodels.index(m) for m in models]]
    coi = cpar.index('operation_span'); cci = cpar.index('cvlt_word_list')
    CLAB = [DM[p] for p in cpar]
    CPAIRS = [(i, j) for i in range(13) for j in range(i + 1, 13)]
    CW = np.array([CLAB[i] == CLAB[j] for i, j in CPAIRS])

    def cdelta(corr):
        v = np.array([corr[i, j] for i, j in CPAIRS])
        return float(v[CW].mean() - v[~CW].mean())

    def cperm_exact(corr):
        globals_backup = (LAB[:], )
        # reuse perm_exact machinery by relabeling through the construct order
        import itertools
        v = {(i, j): corr[i, j] for i, j in CPAIRS}
        idx = list(range(13)); d_obs = cdelta(corr); S_tot = sum(v.values())
        deltas = []
        for nine in itertools.combinations(idx, 9):
            rest4 = [x for x in idx if x not in nine]
            a0 = nine[0]
            for g1r in itertools.combinations(nine[1:], 2):
                g1 = {a0, *g1r}
                rem6 = [x for x in nine if x not in g1]
                b0 = rem6[0]
                for g2r in itertools.combinations(rem6[1:], 2):
                    g2 = {b0, *g2r}
                    g3 = [x for x in rem6 if x not in g2]
                    c0 = rest4[0]
                    for dp in rest4[1:]:
                        h1 = {c0, dp}
                        h2 = [x for x in rest4 if x not in h1]
                        Sw = 0.0
                        for grp in (tuple(sorted(g1)), tuple(sorted(g2)), tuple(sorted(g3))):
                            Sw += v[(grp[0], grp[1])] + v[(grp[0], grp[2])] + v[(grp[1], grp[2])]
                        Sw += v[tuple(sorted(h1))] + v[tuple(sorted(h2))]
                        deltas.append(Sw / 11.0 - (S_tot - Sw) / 67.0)
        ds = np.array(deltas)
        assert len(ds) == 600600
        return d_obs, float((ds >= d_obs).mean()), float((np.abs(ds) >= abs(d_obs)).mean())

    def czscore(X): return (X - X.mean(0)) / X.std(0, ddof=0)

    def cpc1_removed(X):
        Z = czscore(X); U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        return Z - np.outer(U[:, 0] * S[0], Vt[0])

    construct = {}
    ctuples = [('base', None, None), ('aplus_strict', col_s, col_cv),
               ('aplus_canonical', col_c, col_cv)]
    if col_a is not None:
        ctuples.append(('aplus_adjudicated', col_a, col_cv))
    for tag, oc, cc in ctuples:
        X = C.copy()
        if oc is not None:
            X[:, coi] = oc; X[:, cci] = cc
        d, p1, p2 = cperm_exact(np.corrcoef(X.T))
        dr, _, pr2 = cperm_exact(np.corrcoef(cpc1_removed(X).T))
        ev = np.linalg.eigvalsh(np.corrcoef(czscore(X).T))[::-1]
        construct[tag] = {'delta': round(d, 4), 'p2_exact': round(p2, 5),
                          'pc1rm_delta': round(dr, 4), 'pc1rm_p2_exact': round(pr2, 5),
                          'pc1_share': round(float(ev[0] / ev.sum()), 4)}
    results['construct_native'] = construct

    # Frozen construct matrices for the downstream construct branch
    # (validate_pc1 / joint_exclusion consume the selected one through
    # COGARENA_PRIMARY_CONSTRUCT_MATRIX).  Canonical remains a sensitivity.
    construct_matrices = [('matrix_construct_aplus_strict', col_s),
                          ('matrix_construct_aplus_canonical', col_c)]
    if col_a is not None:
        construct_matrices.append(('matrix_construct_aplus_adjudicated', col_a))
    for name, ospan_col in construct_matrices:
        CX = C.copy(); CX[:, coi] = ospan_col; CX[:, cci] = col_cv
        with open(f"{OUT}/{name}.csv", 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['model'] + cpar)
            for mi, m in enumerate(models):
                w.writerow([m] + [f"{x:.5f}" for x in CX[mi]])

    SIZE20 = {'tinyllama:1.1b': 1.1, 'qwen2.5:0.5b': 0.5, 'qwen2.5:1.5b': 1.5, 'gemma2:2b': 2,
              'llama3.2:1b': 1, 'qwen2.5:3b': 3, 'llama3.2:3b': 3, 'qwen2.5:7b': 7, 'mistral:7b': 7,
              'llama3.1:8b': 8, 'deepseek-r1:7b': 7, 'gemma2:9b': 9, 'qwen2.5:14b': 14, 'phi3:14b': 14,
              'deepseek-r1:14b': 14, 'gemma2:27b': 27, 'qwen2.5:32b': 32, 'mixtral:8x7b': 47,
              'yi:34b': 34, 'command-r:35b': 35}
    from scipy import stats as _st
    i20 = [models.index(m) for m in models if m in SIZE20]
    ls20 = np.log([SIZE20[models[i]] for i in i20])
    sizes55 = {}
    for m in models:
        if m in SIZE20:
            sizes55[m] = SIZE20[m]
        else:
            sz = new_meta.get(m, {}).get('params_b') or new_meta.get(m, {}).get('size_b')
            if sz:
                sizes55[m] = float(sz)
    i55 = [models.index(m) for m in models if m in sizes55]
    ls55 = np.log([sizes55[models[i]] for i in i55])

    def rp(lsv, idxs, col):
        r, p = _st.pearsonr(lsv, col[idxs])
        return {'r': round(float(r), 4), 'p': round(float(p), 4), 'n': len(idxs)}
    # 20- and 55-pool claims are deliberately separate; the 55-pool
    # family-aware MixedLM refresh (singular-fit fallback included) is part
    # of the downstream scaling-artifact update, not this suite.
    results['scaling'] = {
        'pool20': {'ospan_production': rp(ls20, i20, M[:, OI]),
                   'ospan_strict': rp(ls20, i20, col_s),
                   'ospan_canonical': rp(ls20, i20, col_c),
                   'cvlt_production': rp(ls20, i20, M[:, CI]),
                   'cvlt_fixed': rp(ls20, i20, col_cv)},
        'pool55': {'ospan_strict': rp(ls55, i55, col_s),
                   'ospan_canonical': rp(ls55, i55, col_c),
                   'cvlt_fixed': rp(ls55, i55, col_cv),
                   'note': 'family-aware mixed-model refresh deferred to scaling artifact update'}}
    if col_a is not None:
        results['scaling']['pool20']['ospan_adjudicated'] = rp(ls20, i20, col_a)
        results['scaling']['pool55']['ospan_adjudicated'] = rp(ls55, i55, col_a)

    # ---- artifacts --------------------------------------------------------- #
    json.dump({m: ospan[m] for m in models}, open(f"{OUT}/ospan_recall_scores.json", 'w'), indent=1)
    json.dump({m: cvlt[m] for m in models}, open(f"{OUT}/cvlt_fixed_scores.json", 'w'), indent=1)
    json.dump({'raw': fam_raw, 'merged': fam_merged}, open(f"{OUT}/family_map.json", 'w'), indent=1)
    if gold and not os.path.exists(GOLD_MANIFEST):
        json.dump({t: g for t, g in sorted(gold.items())}, open(GOLD_MANIFEST, 'w'), indent=1)
    matrices = [('matrix_aplus_strict', MA), ('matrix_aplus_canonical', MC)]
    if col_a is not None:
        matrices += [('matrix_aplus_adjudicated', MJ), ('matrix_aplus_adjudicated_exclamb', MJX)]
    for name, X in matrices:
        with open(f"{OUT}/{name}.csv", 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['model'] + paradigms)
            for i, m in enumerate(models):
                w.writerow([m] + [f"{x:.5f}" for x in X[i]])

    ep_files = [f for m in models for par, dim in
                (('operation_span', 'working_memory'), ('cvlt_word_list', 'episodic_memory'),
                 ('n_back', 'working_memory'))
                for f in ep_paths(m, par, dim)]
    static_files = []
    cc_files = []
    for m in models:
        is_old = m in b2.OLD_MODELS
        s = "full_eval_20260526_2208" if is_old else "full_eval_expansion"
        p = f"{ROOT}/results/{s}/openai_{m}/text/details.json"
        if os.path.exists(p):
            static_files.append(p)
        cc_files += glob.glob(f"{ROOT}/results/{s}/openai_{m}/text/metacognition/confidence_calibration/*.json")
    sm_ov_env = os.environ.get('COGARENA_SM_OVERLAY')
    wager_ov_env = os.environ.get('COGARENA_WAGER_OVERLAY')
    results['inputs'] = {
        'source_revision': os.environ.get('COGARENA_GIT_HEAD'),
        'corrected_matrix_sha256': sha_file(f"{ROOT}/results/recompute_20260703/corrected_matrix.csv"),
        'construct_matrix_sha256': sha_file(f"{ROOT}/results/construct_native_20260711/construct_matrix.csv"),
        # When set, item pools consume the SM overlay; the pool self-check
        # then enforces that corrected_matrix.csv was rebuilt with the same
        # overlay (a mismatched SM cell fails loudly).
        'sm_overlay': ({'path': os.path.relpath(sm_ov_env, ROOT), 'sha256': sha_file(sm_ov_env)}
                       if sm_ov_env else None),
        'wager_overlay': ({'path': os.path.relpath(wager_ov_env, ROOT),
                           'sha256': sha_file(wager_ov_env)}
                          if wager_ov_env else None),
        'gold_manifest_sha256': sha_file(GOLD_MANIFEST) if os.path.exists(GOLD_MANIFEST) else None,
        'snapshot_sha256': sha_file(SNAPSHOT) if os.path.exists(SNAPSHOT) else None,
        'expansion_models_sha256': sha_file(f"{ROOT}/results/reanalysis/expansion_models.json"),
        'conf_cal_corrected_sha256': sha_file(f"{ROOT}/results/reanalysis/conf_cal_corrected.json"),
        'episode_tree_sha256': sha_tree(ep_files), 'episode_tree_n_files': len(ep_files),
        'static_details_tree_sha256': sha_tree(static_files), 'static_details_n_files': len(static_files),
        'cc_response_tree_sha256': sha_tree(cc_files), 'cc_response_n_files': len(cc_files),
        'gonogo_tree_sha256': sha_tree(glob.glob(f"{ROOT}/results/gonogo_rerun_20260702/*/text/details.json")),
        'rescore_overlay_tree_sha256': sha_tree(glob.glob(f"{ROOT}/results/rescore_20260702/new_scores/*.json")),
        'script_sha256': sha_file(os.path.abspath(__file__)),
        'tree_hash_method': 'sorted ROOT-relative path + per-file sha256'}
    json.dump(results, open(f"{OUT}/MANIFEST.json", 'w'), indent=1)
    outs = {os.path.basename(p): sha_file(p)
            for p in sorted(glob.glob(f"{OUT}/*.json") + glob.glob(f"{OUT}/*.csv"))
            if os.path.basename(p) not in ('MANIFEST.json', 'CHAIN_MANIFEST.json')}
    results['outputs_sha256'] = outs
    json.dump(results, open(f"{OUT}/MANIFEST.json", 'w'), indent=1)
    print('math:', results['math_process'])
    print('gates(complete-case):', results['math_gate_sensitivity_complete_case'])
    print('construct:', json.dumps(construct))
    print('scaling:', json.dumps(results['scaling']))
    print('parse status:', status_counts, '| hits>total(designated):', over_production)
    solar = [m for m in models if m.startswith('solar')]
    if solar:
        print('solar ospan strict:', round(float(np.mean([e['strict'] for e in ospan[solar[0]]])), 4))
    print('artifacts written to', OUT)

if __name__ == '__main__':
    main()
