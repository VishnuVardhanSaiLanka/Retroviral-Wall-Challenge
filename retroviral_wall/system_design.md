System Design Document: Mechanistic Gate Simulation for RT Activity Prediction

1. System Overview
1.1 Architecture
Copy┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  rt_sequences.csv │ structures/*.pdb │ handcrafted_features.csv  │
│  esm2_embeddings.npz │ feature_dictionary.csv                    │
└──────────────┬──────────────────────────────────┬────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐    ┌─────────────────────────────────┐
│   EXTERNAL DATA LOADER   │    │    EXISTING FEATURE EXTRACTOR   │
│                          │    │                                 │
│  • PE cryo-EM structure  │    │  • Parse handcrafted_features   │
│  • SpCas9-DNA complex    │    │  • Parse ESMFold pLDDT          │
│  • RT-substrate crystals │    │  • Handle NaN/missingness       │
│  • Known RT processivity │    │                                 │
│    measurements          │    │                                 │
└──────────┬───────────────┘    └───────────┬─────────────────────┘
           │                                │
           ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GATE PIPELINE                              │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │  Gate 1   │ │  Gate 2   │ │  Gate 3   │ │ Gate 4 │ │ Gate 5 │  │
│  │ Foldabil- │ │ Fusion    │ │ Substrate │ │Catalyt-│ │Process-│  │
│  │ ity &     │ │ Compat-   │ │ Binding   │ │ic Com- │ │ivity   │  │
│  │ Stability │ │ ibility   │ │ Groove    │ │petence │ │        │  │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └───┬────┘ └───┬────┘  │
│        │             │             │            │          │       │
│        ▼             ▼             ▼            ▼          ▼       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              GATE SCORE MATRIX  (57 × N_gates)             │  │
│  └────────────────────────┬────────────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION LAYER                              │
│                                                                  │
│  Gate scores + selected handcrafted residuals                    │
│         ↓                                                        │
│  Bayesian Logistic Regression / BART                             │
│         ↓                                                        │
│  LOFO Cross-Validation (primary)                                 │
│  LOO Cross-Validation  (secondary)                               │
│         ↓                                                        │
│  P(active), binary prediction, efficiency ranking                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                │
│                                                                  │
│  predictions.csv:  rt_name, predicted_active, predicted_score    │
│  gate_scores.csv:  rt_name, gate1, gate2, ..., gate5             │
│  report.md:        per-gate analysis, LOFO results, figures      │
└─────────────────────────────────────────────────────────────────┘
1.2 Core Design Principles
Copy1. The 57 samples NEVER train the gates. Gates are computed from 
   external knowledge, physics, and pre-trained models.

2. The 57 samples ONLY calibrate the integration layer (how gate 
   scores combine → final prediction).

3. Each gate produces a continuous score ∈ [0, 1] where 1 = fully 
   passes the gate. Scores are interpretable.

4. Gates are independently testable. Each gate's discriminative 
   power is measured before integration.

5. Failure at any gate → low overall score (multiplicative logic, 
   not averaging).

2. File Structure
Copyretroviral_wall/
│
├── config.py                       # Paths, constants, hyperparams
├── main.py                         # Full pipeline orchestrator
├── download_external.py            # Fetch external PDBs & data
│
├── data/                           # PROVIDED (do not modify)
│   ├── rt_sequences.csv
│   ├── handcrafted_features.csv
│   ├── esm2_embeddings.npz
│   ├── family_splits.csv
│   ├── feature_dictionary.csv
│   └── structures/                 # 57 ESMFold PDBs
│
├── external/                       # DOWNLOADED by pipeline
│   ├── pdb/
│   │   ├── pe2_cryo_em.pdb        # Prime editor 2 structure
│   │   ├── cas9_dna_complex.pdb   # SpCas9 bound to DNA
│   │   ├── mmlv_rt_substrate.pdb  # MMLV-RT with RNA:DNA hybrid
│   │   └── hiv1_rt_substrate.pdb  # HIV-1 RT reference complex
│   └── literature/
│       └── rt_processivity.csv    # Curated from literature
│
├── gates/
│   ├── __init__.py
│   ├── base.py                    # AbstractGate class
│   ├── gate1_foldability.py
│   ├── gate2_fusion.py
│   ├── gate3_substrate_binding.py
│   ├── gate4_catalytic.py
│   └── gate5_processivity.py
│
├── calibration/
│   ├── __init__.py
│   ├── integrator.py              # Combines gates → prediction
│   └── evaluation.py              # LOFO, LOO, metrics
│
├── utils/
│   ├── __init__.py
│   ├── pdb_utils.py               # PDB parsing, coordinate extraction
│   ├── geometry.py                # Distance, angle, clash calculations
│   ├── electrostatics.py         # Charge surface computation
│   ├── alignment.py              # Structural alignment wrappers
│   └── io_utils.py               # Data loading helpers
│
├── analysis/
│   ├── gate_diagnostics.py        # Per-gate discriminative power
│   ├── feature_importance.py      # What drives each gate
│   └── visualisation.py           # Plots for writeup
│
├── notebooks/
│   ├── 01_eda_and_existing_features.ipynb
│   ├── 02_gate_development.ipynb
│   ├── 03_calibration_and_results.ipynb
│   └── 04_ablation_studies.ipynb
│
├── outputs/
│   ├── gate_scores/               # Per-gate scores
│   ├── predictions/               # Final predictions
│   ├── figures/                   # For writeup
│   └── logs/                      # Run logs
│
├── tests/
│   ├── test_gates.py
│   ├── test_calibration.py
│   └── test_known_rts.py          # Sanity checks on MMLV, HIV-1 RT
│
├── requirements.txt
├── Dockerfile                     # Reproducibility
└── README.md

3. External Data Specification
3.1 Required PDB Structures
pythonCopy# config.py

EXTERNAL_PDBS = {
    # Prime editor 2 cryo-EM structure (Cas9-MMLV fusion with pegRNA and DNA)
    # Source: Protein Data Bank
    "pe2_complex": {
        "pdb_id": "8W8H",  # or latest available PE structure
        "description": "PE2 nCas9-MMLV-RT with pegRNA and target DNA",
        "usage": "Gate 2 - defines fusion geometry and RT positioning",
        "chains": {
            "cas9": "A",
            "mmlv_rt": "B",      # or wherever RT is
            "pegrna": "C",
            "target_dna": "D,E"
        }
    },
    
    # MMLV-RT with RNA:DNA hybrid (for substrate binding reference)
    "mmlv_rt_substrate": {
        "pdb_id": "7UVO",  # or available MMLV-RT complex
        "description": "MMLV-RT bound to template-primer",
        "usage": "Gate 3 - reference binding groove geometry"
    },
    
    # HIV-1 RT with substrate (backup reference, very well studied)
    "hiv1_rt_substrate": {
        "pdb_id": "1RTD",
        "description": "HIV-1 RT with RNA:DNA hybrid",
        "usage": "Gate 3 - secondary reference"
    },

    # SpCas9 alone (for clash analysis)
    "cas9_alone": {
        "pdb_id": "4ZT0",
        "description": "SpCas9 crystal structure",
        "usage": "Gate 2 - Cas9 body for clash detection"
    }
}
3.2 Literature-Curated RT Data
pythonCopy# external/literature/rt_processivity.csv
# Manually curated from published biochemistry papers
#
# Columns:
#   rt_name: identifier (match to dataset where possible)
#   organism: source organism
#   processivity_nt: nucleotides per binding event (if known)
#   kcat_s: turnover number (if known)
#   optimal_temp_C: optimal temperature
#   source_doi: publication DOI
#
# Target: ~15-30 RTs with quantitative processivity data
# Key sources:
#   - Gerard et al. (1986) - MMLV processivity
#   - Huber et al. (1989) - AMV-RT
#   - Bibillo & Eickbush (2002) - Group II intron RTs
#   - Lamprea-Burgunder et al. (2023) - retron RTs
3.3 Download Script
pythonCopy# download_external.py

import os
import urllib.request

PDB_BASE_URL = "https://files.rcsb.org/download"

def download_pdb(pdb_id: str, output_dir: str = "external/pdb") -> str:
    """Download PDB file from RCSB."""
    os.makedirs(output_dir, exist_ok=True)
    url = f"{PDB_BASE_URL}/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if not os.path.exists(output_path):
        print(f"Downloading {pdb_id}...")
        urllib.request.urlretrieve(url, output_path)
    
    return output_path

def download_all_external():
    """Download all required external structures."""
    from config import EXTERNAL_PDBS
    
    for name, info in EXTERNAL_PDBS.items():
        path = download_pdb(info["pdb_id"])
        print(f"  {name}: {path}")

if __name__ == "__main__":
    download_all_external()

4. Gate Specifications
4.0 Abstract Base Class
pythonCopy# gates/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class GateResult:
    """Output of a single gate for a single RT."""
    rt_name: str
    gate_name: str
    score: float              # ∈ [0, 1], higher = more likely to pass
    sub_scores: dict          # Component scores for interpretability
    confidence: float         # How reliable is this score? ∈ [0, 1]
    failure_reason: str       # Human-readable if score is low

class AbstractGate(ABC):
    """
    Base class for all mechanistic gates.
    
    Contract:
    - Gates do NOT see activity labels during score computation
    - Gates use external data, physics, and pre-trained models only
    - Gates produce interpretable scores
    """
    
    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def compute_scores(
        self,
        sequences: pd.DataFrame,       # rt_sequences.csv loaded
        structures_dir: str,            # path to structures/
        handcrafted: pd.DataFrame,      # handcrafted_features.csv loaded
        external_data: dict             # loaded external PDBs etc.
    ) -> list[GateResult]:
        """Compute gate score for all 57 RTs. No labels used."""
        pass
    
    def score_matrix(self, results: list[GateResult]) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        rows = []
        for r in results:
            row = {"rt_name": r.rt_name, f"{self.name}_score": r.score,
                   f"{self.name}_confidence": r.confidence}
            for k, v in r.sub_scores.items():
                row[f"{self.name}_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

4.1 Gate 1: Foldability & Stability
Biological question: Does this RT fold into a stable, soluble protein at 37°C?
Why it matters: An RT that misfolds or aggregates in mammalian cells never gets a chance to catalyse anything. This gate eliminates sequences that are fundamentally unviable as proteins.
Data sources: Existing handcrafted features + ESMFold pLDDT from structure files. No external training needed. Pure physics/bioinformatics.
pythonCopy# gates/gate1_foldability.py

import numpy as np
from Bio.PDB import PDBParser
from gates.base import AbstractGate, GateResult

class FoldabilityGate(AbstractGate):
    """
    Scores RT foldability and thermostability.
    
    Sub-scores:
      - plddt_mean:     Mean pLDDT from ESMFold structure (0-1)
      - plddt_core:     pLDDT of core residues (not termini) (0-1)
      - thermo_37:      Predicted fraction folded at 37°C (from features)
      - solubility:     CamSol intrinsic solubility score (normalised)
      - instability:    ProtParam instability index (inverted, normalised)
    
    Aggregation: Weighted geometric mean (multiplicative gate logic)
    """
    
    def __init__(self):
        super().__init__("foldability")
        
        # Weights reflect relative importance
        # (these are FIXED, not learned from the 57 samples)
        self.weights = {
            "plddt_mean": 0.30,
            "plddt_core": 0.20,
            "thermo_37": 0.25,
            "solubility": 0.15,
            "instability": 0.10,
        }
    
    def _extract_plddt(self, pdb_path: str) -> tuple[float, float]:
        """
        Extract mean pLDDT and core pLDDT from ESMFold PDB.
        ESMFold stores pLDDT in the B-factor column.
        Core = exclude first/last 20 residues.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rt", pdb_path)
        
        b_factors = []
        for model in structure:
            for chain in model:
                residues = list(chain.get_residues())
                for i, res in enumerate(residues):
                    for atom in res:
                        if atom.name == "CA":
                            b_factors.append({
                                "residue_idx": i,
                                "plddt": atom.get_bfactor(),
                                "is_core": 20 <= i <= len(residues) - 20
                            })
        
        all_plddt = [b["plddt"] for b in b_factors]
        core_plddt = [b["plddt"] for b in b_factors if b["is_core"]]
        
        mean_plddt = np.mean(all_plddt) / 100.0 if all_plddt else 0.0
        mean_core = np.mean(core_plddt) / 100.0 if core_plddt else mean_plddt
        
        return mean_plddt, mean_core
    
    def _normalise_solubility(self, camsol_score: float) -> float:
        """
        CamSol intrinsic solubility: higher = more soluble.
        Typical range for globular proteins: -2 to +2.
        Map to [0, 1] with sigmoid centered at 0.
        """
        return 1.0 / (1.0 + np.exp(-camsol_score))
    
    def _normalise_instability(self, instability_index: float) -> float:
        """
        ProtParam instability index: < 40 = stable, > 40 = unstable.
        Invert and normalise: lower instability → higher score.
        """
        return 1.0 / (1.0 + np.exp((instability_index - 40) / 10))
    
    def compute_scores(self, sequences, structures_dir, handcrafted, 
                       external_data) -> list[GateResult]:
        results = []
        
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            
            # --- Sub-score 1 & 2: pLDDT ---
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            plddt_mean, plddt_core = self._extract_plddt(pdb_path)
            
            # --- Sub-score 3: Thermostability at 37°C ---
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            # Column name from feature_dictionary - adjust as needed
            thermo_37 = hc.get("thermo_frac_folded_37C", np.nan)
            if np.isnan(thermo_37):
                thermo_37 = 0.5  # Neutral prior if missing
            
            # --- Sub-score 4: Solubility ---
            camsol = hc.get("camsol_intrinsic_solubility", np.nan)
            solubility = self._normalise_solubility(camsol) if not np.isnan(camsol) else 0.5
            
            # --- Sub-score 5: Instability ---
            instab = hc.get("protparam_instability_index", np.nan)
            instability_score = self._normalise_instability(instab) if not np.isnan(instab) else 0.5
            
            sub_scores = {
                "plddt_mean": plddt_mean,
                "plddt_core": plddt_core,
                "thermo_37": thermo_37,
                "solubility": solubility,
                "instability": instability_score,
            }
            
            # Weighted geometric mean (multiplicative: weak link dominates)
            log_score = sum(
                self.weights[k] * np.log(max(v, 1e-6))
                for k, v in sub_scores.items()
            )
            score = np.exp(log_score)
            
            # Confidence: high if we have all data
            n_missing = sum(1 for v in sub_scores.values() if v == 0.5)
            confidence = 1.0 - (n_missing * 0.15)
            
            failure_reason = ""
            if plddt_mean < 0.5:
                failure_reason = "Poor ESMFold confidence suggests misfolding"
            elif thermo_37 < 0.3:
                failure_reason = "Predicted unstable at 37°C"
            
            results.append(GateResult(
                rt_name=rt_name,
                gate_name=self.name,
                score=float(np.clip(score, 0, 1)),
                sub_scores=sub_scores,
                confidence=confidence,
                failure_reason=failure_reason
            ))
        
        return results
Diagnostic check: After computing Gate 1 scores, verify that MMLV-RT (the gold standard, 41% PE efficiency) gets a high score and that some known inactives get low scores. If Gate 1 doesn't discriminate at all (all scores similar), it's not a useful gate for this dataset — but it's still meaningful as a prerequisite filter.

4.2 Gate 2: Fusion Compatibility with Cas9
Biological question: When fused to Cas9 nickase via a flexible linker, does the RT (a) avoid steric clash with Cas9 and the DNA, (b) present its active site in an accessible orientation, and (c) physically fit in the available space?
Why this is probably the key gate: Every RT in the dataset was tested as a Cas9 fusion. An RT might be catalytically perfect in isolation but completely incompatible with the fusion geometry. This constraint is invisible to any model that analyses the RT in isolation.
Strategy: Use the known PE2 cryo-EM structure to define the spatial context. Superimpose each candidate RT onto MMLV-RT's position in the complex. Score steric compatibility.
pythonCopy# gates/gate2_fusion.py

import numpy as np
from Bio.PDB import PDBParser, Superimposer, NeighborSearch
from gates.base import AbstractGate, GateResult
from utils.alignment import structural_align
from utils.geometry import (
    compute_steric_clashes,
    compute_active_site_accessibility,
    extract_ca_coords,
    get_terminus_flexibility,
)

class FusionCompatibilityGate(AbstractGate):
    """
    Scores compatibility of RT with the Cas9 fusion context.
    
    Approach:
    1. Load PE2 cryo-EM structure (Cas9 + MMLV-RT + DNA + pegRNA)
    2. Extract MMLV-RT coordinates and Cas9+DNA coordinates
    3. For each candidate RT:
       a. Structurally align candidate to MMLV-RT (TM-align)
       b. Place candidate in MMLV-RT's position within the complex
       c. Score steric clashes with Cas9 body and DNA
       d. Score active site accessibility (is it pointing the right way?)
       e. Score size compatibility (very large RTs = more clashes)
       f. Score terminus flexibility (rigid N-term = bad for linker)
    
    Sub-scores:
      - clash_score:        Inverse of # atoms clashing with Cas9/DNA (0-1)
      - alignment_quality:  TM-score of candidate vs MMLV-RT (0-1)
      - active_site_access: Active site points toward DNA? (0-1)
      - size_penalty:       Larger RT = more potential for clash (0-1)
      - terminus_flex:      N-terminal pLDDT (flexible = good for linker)
    """
    
    def __init__(self):
        super().__init__("fusion_compat")
        
        self.weights = {
            "clash_score": 0.30,
            "alignment_quality": 0.15,
            "active_site_access": 0.25,
            "size_penalty": 0.15,
            "terminus_flex": 0.15,
        }
        
        # Steric clash threshold (Angstroms)
        self.CLASH_DISTANCE = 2.0     # Hard clash
        self.CONTACT_DISTANCE = 4.0   # Close contact
        
        # Maximum RT size before penalty kicks in
        self.MAX_SIZE_NO_PENALTY = 800  # aa (MMLV is ~700)
    
    def _load_pe2_context(self, external_data: dict):
        """
        Parse PE2 cryo-EM structure.
        Extract:
          - MMLV-RT atom coordinates (for alignment target)
          - Cas9 + DNA atom coordinates (for clash detection)
          - MMLV-RT active site residue positions
        """
        parser = PDBParser(QUIET=True)
        pe2_path = external_data["pe2_complex"]["path"]
        structure = parser.get_structure("pe2", pe2_path)
        
        # Chain assignments - ADJUST based on actual PDB
        pe2_info = external_data["pe2_complex"]
        
        # You'll need to inspect the PDB to identify:
        # - Which chain(s) = Cas9
        # - Which chain = RT
        # - Which chain(s) = nucleic acids
        
        # Store as instance variables
        self.mmlv_rt_atoms = self._get_chain_atoms(structure, pe2_info["chains"]["mmlv_rt"])
        self.cas9_atoms = self._get_chain_atoms(structure, pe2_info["chains"]["cas9"])
        self.dna_atoms = self._get_chain_atoms(structure, pe2_info["chains"]["target_dna"])
        
        # Combined "environment" atoms (everything except MMLV-RT)
        self.env_atoms = self.cas9_atoms + self.dna_atoms
        self.env_coords = np.array([a.get_vector().get_array() for a in self.env_atoms])
        
        # MMLV-RT active site (catalytic Asp residues)
        # Identify by known MMLV catalytic residues: D150, D224, D225
        # (adjust residue numbers based on actual PDB numbering)
        self.mmlv_active_site_center = self._get_active_site_center(
            structure, pe2_info["chains"]["mmlv_rt"],
            catalytic_residues=[150, 224, 225]  # ADJUST
        )
    
    def _get_chain_atoms(self, structure, chain_ids):
        """Extract all atoms from specified chain(s)."""
        atoms = []
        chain_ids = chain_ids.split(",") if "," in chain_ids else [chain_ids]
        for model in structure:
            for chain in model:
                if chain.id in chain_ids:
                    for residue in chain:
                        for atom in residue:
                            atoms.append(atom)
        return atoms
    
    def _get_active_site_center(self, structure, chain_id, catalytic_residues):
        """Get center of mass of catalytic residues."""
        coords = []
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.id[1] in catalytic_residues:
                            for atom in residue:
                                if atom.name == "CA":
                                    coords.append(atom.get_vector().get_array())
        return np.mean(coords, axis=0) if coords else None
    
    def _score_single_rt(self, rt_name, rt_pdb_path, sequence_length):
        """
        Score one candidate RT for fusion compatibility.
        
        Pipeline:
        1. Structurally align candidate to MMLV-RT
        2. Apply the transformation to place candidate in PE2 context
        3. Count clashes with Cas9/DNA environment
        4. Assess active site orientation
        """
        
        # Step 1: Structural alignment to MMLV-RT
        # Using TM-align or BioPython Superimposer
        alignment_result = structural_align(
            mobile_pdb=rt_pdb_path,
            target_atoms=self.mmlv_rt_atoms,
            method="tmalign"
        )
        # alignment_result contains:
        #   .tm_score: float
        #   .rotation: 3x3 matrix
        #   .translation: 3x1 vector
        #   .aligned_atoms: transformed coordinates
        
        if alignment_result.tm_score < 0.2:
            # Can't meaningfully align - too structurally different
            return {
                "clash_score": 0.3,  # Uncertain, give weak default
                "alignment_quality": alignment_result.tm_score,
                "active_site_access": 0.3,
                "size_penalty": self._size_score(sequence_length),
                "terminus_flex": 0.5,
            }, f"Poor structural alignment to MMLV-RT (TM={alignment_result.tm_score:.2f})"
        
        # Step 2: Get transformed candidate coordinates
        transformed_coords = alignment_result.aligned_atoms  # Nx3 array
        
        # Step 3: Count steric clashes
        n_clashes, n_contacts = compute_steric_clashes(
            query_coords=transformed_coords,
            env_coords=self.env_coords,
            clash_dist=self.CLASH_DISTANCE,
            contact_dist=self.CONTACT_DISTANCE,
        )
        
        # Normalise: more clashes = lower score
        # Scale: 0 clashes → 1.0, 50+ clashes → ~0.0
        clash_score = np.exp(-n_clashes / 20.0)
        
        # Step 4: Active site accessibility
        # After alignment, where is the candidate's active site?
        # It should point toward the DNA (where the 3'OH primer is)
        candidate_active_site = alignment_result.active_site_center_transformed
        
        if candidate_active_site is not None and self.mmlv_active_site_center is not None:
            # Distance between candidate active site and MMLV active site position
            # (in the PE2 context - closer = better)
            as_distance = np.linalg.norm(
                candidate_active_site - self.mmlv_active_site_center
            )
            active_site_access = np.exp(-as_distance / 10.0)  # 10Å scale
        else:
            active_site_access = 0.3  # Can't assess
        
        # Step 5: Size penalty
        size_score = self._size_score(sequence_length)
        
        # Step 6: Terminus flexibility
        # (compute from pLDDT of first 10 residues of candidate)
        terminus_flex = get_terminus_flexibility(rt_pdb_path, n_residues=10)
        # Low pLDDT at N-terminus = flexible = good for linker
        # Invert: flexible → high score
        terminus_flex_score = 1.0 - terminus_flex  # terminus_flex is mean pLDDT/100
        
        sub_scores = {
            "clash_score": clash_score,
            "alignment_quality": alignment_result.tm_score,
            "active_site_access": active_site_access,
            "size_penalty": size_score,
            "terminus_flex": terminus_flex_score,
        }
        
        failure = ""
        if n_clashes > 30:
            failure = f"Severe steric clash ({n_clashes} atoms) with Cas9/DNA"
        elif active_site_access < 0.3:
            failure = "Active site poorly oriented in fusion context"
        
        return sub_scores, failure
    
    def _size_score(self, length):
        """Penalty for very large or very small RTs."""
        if length <= self.MAX_SIZE_NO_PENALTY:
            return 1.0
        else:
            return np.exp(-(length - self.MAX_SIZE_NO_PENALTY) / 200.0)
    
    def compute_scores(self, sequences, structures_dir, handcrafted, 
                       external_data) -> list[GateResult]:
        
        # Load PE2 context (once)
        self._load_pe2_context(external_data)
        
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            seq_len = row["protein_length_aa"]
            
            sub_scores, failure = self._score_single_rt(rt_name, pdb_path, seq_len)
            
            # Weighted geometric mean
            log_score = sum(
                self.weights[k] * np.log(max(v, 1e-6))
                for k, v in sub_scores.items()
            )
            score = float(np.clip(np.exp(log_score), 0, 1))
            
            # Confidence: high alignment quality = more confident
            confidence = min(sub_scores["alignment_quality"] * 1.5, 1.0)
            
            results.append(GateResult(
                rt_name=rt_name,
                gate_name=self.name,
                score=score,
                sub_scores=sub_scores,
                confidence=confidence,
                failure_reason=failure
            ))
        
        return results
Critical implementation note: The PE2 cryo-EM structure might not cleanly separate Cas9 and RT chains. You may need to manually inspect the PDB and define chain boundaries. Also, TM-align is an external binary — install it or use the tmtools Python package.
Fallback if PE2 structure is unavailable or insufficient: Model the fusion context more simply:

Take the Cas9-DNA structure alone
Place the RT at Cas9's C-terminus with a flexible linker
Sample linker conformations (e.g., with short MC simulation)
Score the minimum-clash orientation


4.3 Gate 3: Substrate Binding Groove
Biological question: Can this RT bind an RNA:DNA hybrid in the template-primer groove?
pythonCopy# gates/gate3_substrate_binding.py

import numpy as np
from gates.base import AbstractGate, GateResult
from utils.geometry import (
    compute_groove_dimensions,
    compute_surface_electrostatics,
)
from utils.alignment import structural_align

class SubstrateBindingGate(AbstractGate):
    """
    Scores RT's ability to bind RNA:DNA hybrid substrate.
    
    Strategy:
    1. Identify the template-primer binding groove in each RT structure
       by structural alignment to a known RT-substrate complex
    2. Measure groove geometry (width, depth, length)
    3. Compute electrostatic potential in the groove (should be positive)
    4. [Optional] Dock RNA:DNA hybrid and score binding energy
    
    Sub-scores:
      - groove_detected:    Was a groove found? (binary → 0 or 1)
      - groove_geometry:    Width/depth similar to functional RTs (0-1)
      - groove_charge:      Electrostatic complementarity (0-1)
      - sasa_active_site:   Active site accessible? (0-1)
      - thumb_contact:      Thumb domain can contact substrate? (0-1)
    """
    
    def __init__(self):
        super().__init__("substrate_binding")
        
        self.weights = {
            "groove_detected": 0.25,
            "groove_geometry": 0.25,
            "groove_charge": 0.20,
            "sasa_active_site": 0.15,
            "thumb_contact": 0.15,
        }
        
        # Reference groove dimensions from MMLV-RT crystal structure
        # (measured from known RT-substrate complexes)
        self.REF_GROOVE_WIDTH = 18.0   # Angstroms (approximate)
        self.REF_GROOVE_DEPTH = 12.0   # Angstroms (approximate)
    
    def _identify_groove_residues(self, rt_pdb_path, reference_complex):
        """
        Identify which residues in the candidate RT correspond to the
        template-primer binding groove.
        
        Method:
        1. Align candidate to MMLV-RT from the reference complex
        2. Map the groove-lining residues from reference to candidate
           via the structural alignment
        3. Extract those residues' coordinates
        
        This avoids sequence-based identification (which fails across families).
        """
        align_result = structural_align(
            mobile_pdb=rt_pdb_path,
            target_pdb=reference_complex,
            method="tmalign"
        )
        
        if align_result.tm_score < 0.25:
            return None, align_result.tm_score  # Can't identify groove
        
        # Known groove-lining residues in MMLV-RT (from crystal structure)
        # These are residues within 6Å of the RNA:DNA hybrid
        # in the reference complex
        mmlv_groove_residues = reference_complex.get_substrate_contacts(
            distance_cutoff=6.0
        )
        
        # Map to candidate via alignment correspondence
        candidate_groove_residues = align_result.map_residues(
            mmlv_groove_residues
        )
        
        return candidate_groove_residues, align_result.tm_score
    
    def _compute_groove_geometry(self, groove_residues, structure):
        """
        Measure groove dimensions from identified groove residues.
        
        Returns normalised score: 1.0 if dimensions match reference,
        decreasing as they diverge.
        """
        if groove_residues is None:
            return 0.0
        
        coords = np.array([
            atom.get_vector().get_array()
            for res in groove_residues
            for atom in res if atom.name == "CA"
        ])
        
        if len(coords) < 5:
            return 0.0
        
        # PCA to estimate groove dimensions
        centered = coords - coords.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        
        # s[0] ≈ groove length, s[1] ≈ width, s[2] ≈ depth
        est_width = s[1] * 2 / np.sqrt(len(coords))  # Scale factor
        est_depth = s[2] * 2 / np.sqrt(len(coords))
        
        # Score: Gaussian penalty for deviation from reference
        width_dev = (est_width - self.REF_GROOVE_WIDTH) / 5.0
        depth_dev = (est_depth - self.REF_GROOVE_DEPTH) / 5.0
        
        geometry_score = np.exp(-0.5 * (width_dev**2 + depth_dev**2))
        
        return float(geometry_score)
    
    def _compute_groove_charge(self, groove_residues):
        """
        Compute net charge of groove-lining residues.
        RNA:DNA hybrid is highly negative → groove should be positive.
        
        Simple approach: count Arg + Lys - Asp - Glu in groove.
        Normalise by groove size.
        """
        if groove_residues is None:
            return 0.5  # No data
        
        positive = {"ARG", "LYS"}
        negative = {"ASP", "GLU"}
        
        n_pos = sum(1 for r in groove_residues if r.get_resname() in positive)
        n_neg = sum(1 for r in groove_residues if r.get_resname() in negative)
        n_total = len(groove_residues)
        
        if n_total == 0:
            return 0.5
        
        # Net charge density
        net_charge_density = (n_pos - n_neg) / n_total
        
        # Positive charge density is good. Map to [0, 1]
        # Typical range: -0.2 to +0.3
        return float(1.0 / (1.0 + np.exp(-net_charge_density * 10)))
    
    def compute_scores(self, sequences, structures_dir, handcrafted,
                       external_data) -> list[GateResult]:
        
        # Load reference RT-substrate complex
        ref_complex = self._load_reference_complex(external_data)
        
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            
            # Identify groove
            groove_residues, tm_score = self._identify_groove_residues(
                pdb_path, ref_complex
            )
            
            groove_detected = 1.0 if groove_residues is not None else 0.0
            groove_geometry = self._compute_groove_geometry(groove_residues, pdb_path)
            groove_charge = self._compute_groove_charge(groove_residues)
            
            # SASA of active site (from handcrafted features)
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            sasa_active = hc.get("sasa_active_site_residues", np.nan)
            sasa_score = float(1.0 / (1.0 + np.exp(-(sasa_active - 50) / 20))) \
                         if not np.isnan(sasa_active) else 0.5
            
            # Thumb domain contact potential (from handcrafted features)
            thumb_detected = not np.isnan(hc.get("thumb_surface_charge", np.nan))
            thumb_score = 0.7 if thumb_detected else 0.2
            
            sub_scores = {
                "groove_detected": groove_detected,
                "groove_geometry": groove_geometry,
                "groove_charge": groove_charge,
                "sasa_active_site": sasa_score,
                "thumb_contact": thumb_score,
            }
            
            # Aggregate
            log_score = sum(
                self.weights[k] * np.log(max(v, 1e-6))
                for k, v in sub_scores.items()
            )
            score = float(np.clip(np.exp(log_score), 0, 1))
            
            confidence = min(tm_score * 2, 1.0) if groove_residues else 0.3
            
            failure = ""
            if groove_detected == 0.0:
                failure = "No recognisable substrate binding groove"
            elif groove_charge < 0.3:
                failure = "Groove is negatively charged (repels substrate)"
            
            results.append(GateResult(
                rt_name=rt_name, gate_name=self.name, score=score,
                sub_scores=sub_scores, confidence=confidence,
                failure_reason=failure
            ))
        
        return results

4.4 Gate 4: Catalytic Competence
Biological question: Is the polymerase catalytic machinery intact and correctly positioned?
pythonCopy# gates/gate4_catalytic.py

import numpy as np
from gates.base import AbstractGate, GateResult

class CatalyticCompetenceGate(AbstractGate):
    """
    Scores integrity of the RT catalytic machinery.
    
    Key requirement: Two-metal-ion mechanism with conserved Asp residues.
    The YXDD (or FADD, YADD, etc.) motif must be present and correctly
    positioned in 3D space.
    
    Sub-scores:
      - motif_present:      Is a catalytic motif detectable? (0 or 1)
      - triad_geometry:     Are catalytic Asp residues at correct distances? (0-1)
      - motif_ss:           Is the motif in the correct secondary structure? (0-1)
      - metal_coordination: Can two metals be coordinated? (0-1)
      - esm_if_perplexity:  Inverse folding score (lower = better fold) (0-1)
    
    Data sources: Primarily handcrafted features (Asp Triad, Motif SS, 
    ESM-IF). Supplemented by direct structure analysis for metal coordination.
    """
    
    def __init__(self):
        super().__init__("catalytic")
        
        self.weights = {
            "motif_present": 0.30,
            "triad_geometry": 0.30,
            "motif_ss": 0.15,
            "metal_coordination": 0.15,
            "esm_if_quality": 0.10,
        }
        
        # Reference Asp triad distances from MMLV-RT crystal structure (Å)
        # D150-D224: ~6.5Å, D224-D225: ~3.8Å (Cα distances)
        self.REF_TRIAD_D1_D2 = 6.5
        self.REF_TRIAD_D2_D3 = 3.8
        self.DISTANCE_TOLERANCE = 3.0  # Å
    
    def compute_scores(self, sequences, structures_dir, handcrafted,
                       external_data) -> list[GateResult]:
        results = []
        
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            
            # --- Motif present ---
            # Infer from whether triad features are NaN
            # NaN in asp_triad features = motif not found = bad
            triad_d1_d2 = hc.get("asp_triad_d1_d2_distance", np.nan)
            triad_d2_d3 = hc.get("asp_triad_d2_d3_distance", np.nan)
            motif_present = 0.0 if (np.isnan(triad_d1_d2) and np.isnan(triad_d2_d3)) else 1.0
            
            # --- Triad geometry ---
            if motif_present > 0:
                dev1 = abs(triad_d1_d2 - self.REF_TRIAD_D1_D2) if not np.isnan(triad_d1_d2) else self.DISTANCE_TOLERANCE
                dev2 = abs(triad_d2_d3 - self.REF_TRIAD_D2_D3) if not np.isnan(triad_d2_d3) else self.DISTANCE_TOLERANCE
                triad_geometry = np.exp(
                    -(dev1**2 + dev2**2) / (2 * self.DISTANCE_TOLERANCE**2)
                )
            else:
                triad_geometry = 0.0
            
            # --- Motif secondary structure ---
            # The catalytic motif should be in a beta-turn/loop
            # (specific features from handcrafted - adjust column names)
            motif_ss_score = hc.get("motif_ss_correct_fraction", np.nan)
            if np.isnan(motif_ss_score):
                motif_ss_score = 0.5 if motif_present else 0.1
            
            # --- Metal coordination ---
            # Proxy: Are there Asp/Glu residues at appropriate distances
            # for two-metal coordination?
            # This requires direct structure analysis
            metal_score = self._assess_metal_coordination(
                f"{structures_dir}/{rt_name}.pdb", hc
            )
            
            # --- ESM-IF perplexity ---
            esm_if = hc.get("esm_if_perplexity", np.nan)
            if not np.isnan(esm_if):
                # Lower perplexity = better folded = higher score
                # Typical range: 3-15 for well-folded, 15+ for poor
                esm_if_quality = np.exp(-max(esm_if - 5, 0) / 10.0)
            else:
                esm_if_quality = 0.5
            
            sub_scores = {
                "motif_present": motif_present,
                "triad_geometry": float(triad_geometry),
                "motif_ss": float(motif_ss_score),
                "metal_coordination": float(metal_score),
                "esm_if_quality": float(esm_if_quality),
            }
            
            log_score = sum(
                self.weights[k] * np.log(max(v, 1e-6))
                for k, v in sub_scores.items()
            )
            score = float(np.clip(np.exp(log_score), 0, 1))
            
            failure = ""
            if motif_present == 0.0:
                failure = "No catalytic motif detected"
            elif triad_geometry < 0.3:
                failure = "Catalytic triad geometry deviates from functional reference"
            
            results.append(GateResult(
                rt_name=rt_name, gate_name=self.name, score=score,
                sub_scores=sub_scores,
                confidence=0.9 if motif_present else 0.4,
                failure_reason=failure
            ))
        
        return results
    
    def _assess_metal_coordination(self, pdb_path, handcrafted_row):
        """
        Check for two-metal-ion coordination geometry.
        
        In RTs, two Mg²⁺ ions are coordinated by the three catalytic Asp
        residues. Metal A is coordinated by Asp1 and Asp3. Metal B by
        all three. Metals are ~3.5-4.0 Å apart.
        
        We check: are there 3 Asp/Glu residues within a ~8Å sphere
        with geometry compatible with two-metal coordination?
        """
        # Simplified: use triad distances as proxy
        # If triad is detected and geometry is reasonable, assume
        # metal coordination is possible
        d1 = handcrafted_row.get("asp_triad_d1_d2_distance", np.nan)
        d2 = handcrafted_row.get("asp_triad_d2_d3_distance", np.nan)
        
        if np.isnan(d1) or np.isnan(d2):
            return 0.2
        
        # Both distances should be < 10Å for metal coordination
        if d1 < 10 and d2 < 10:
            return 0.8
        else:
            return 0.3

4.5 Gate 5: Processivity
Biological question: Can this RT synthesise enough contiguous DNA (10-50 nt) to write the desired edit?
pythonCopy# gates/gate5_processivity.py

import numpy as np
from gates.base import AbstractGate, GateResult

class ProcessivityGate(AbstractGate):
    """
    Scores RT processivity — ability to synthesize extended DNA stretches.
    
    Key determinant: thumb subdomain. The thumb clamps the template-primer
    duplex and prevents dissociation during synthesis.
    
    Sub-scores:
      - thumb_detected:      Is a thumb domain identifiable? (0 or 1)
      - thumb_charge:        Thumb surface charge (should be positive) (0-1)
      - thumb_size:          Thumb subdomain size relative to reference (0-1)
      - structural_sim_mmlv: TM-score to MMLV (high processivity ref) (0-1)
      - structural_sim_hiv:  TM-score to HIV-1 RT (known processivity) (0-1)
      - hairpin_present:     Beta-hairpin near active site (processivity aid) (0-1)
      - protein_length_norm: Longer RTs tend to have larger thumb domains (0-1)
    
    Literature data integration:
      If the RT has known processivity measurements from literature,
      use those directly (scaled to 0-1).
    """
    
    def __init__(self):
        super().__init__("processivity")
        
        self.weights = {
            "thumb_detected": 0.25,
            "thumb_quality": 0.20,  # combines charge + size
            "structural_sim": 0.25, # best TM-score to processive reference
            "hairpin": 0.15,
            "length_proxy": 0.15,
        }
        
        self.MIN_PROCESSIVE_LENGTH = 200   # aa - very short RTs likely non-processive
        self.IDEAL_LENGTH_RANGE = (300, 700)  # aa
    
    def compute_scores(self, sequences, structures_dir, handcrafted,
                       external_data) -> list[GateResult]:
        
        # Load literature processivity data (if available)
        lit_processivity = external_data.get("rt_processivity", {})
        
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            seq_len = row["protein_length_aa"]
            
            # --- Thumb domain ---
            thumb_charge = hc.get("thumb_surface_charge", np.nan)
            thumb_detected = 0.0 if np.isnan(thumb_charge) else 1.0
            
            # Thumb quality: positive charge = good (grips neg. DNA)
            if thumb_detected:
                # Normalise: positive charge → high score
                thumb_quality = float(1.0 / (1.0 + np.exp(-thumb_charge / 5.0)))
            else:
                thumb_quality = 0.15
            
            # --- Structural similarity to processive RTs ---
            # Use FoldSeek TM-scores from handcrafted features
            tm_mmlv = hc.get("foldseek_TM_MMLV", 0.0)
            tm_hiv = hc.get("foldseek_TM_HIV1RT", 0.0)
            if np.isnan(tm_mmlv): tm_mmlv = 0.0
            if np.isnan(tm_hiv): tm_hiv = 0.0
            
            # Best structural similarity to any known processive RT
            structural_sim = max(tm_mmlv, tm_hiv)
            
            # --- Beta-hairpin ---
            hairpin_features = [c for c in handcrafted.columns if "hairpin" in c.lower()]
            hairpin_detected = any(
                hc.get(f, 0) > 0 for f in hairpin_features
                if not np.isnan(hc.get(f, np.nan))
            )
            hairpin_score = 0.7 if hairpin_detected else 0.3
            
            # --- Length proxy ---
            if seq_len < self.MIN_PROCESSIVE_LENGTH:
                length_score = 0.1
            elif self.IDEAL_LENGTH_RANGE[0] <= seq_len <= self.IDEAL_LENGTH_RANGE[1]:
                length_score = 0.9
            else:
                # Longer than ideal: slight penalty (harder to package but 
                # may still be processive)
                length_score = max(0.4, 0.9 * np.exp(
                    -(seq_len - self.IDEAL_LENGTH_RANGE[1]) / 500
                ))
            
            # --- Literature override ---
            if rt_name in lit_processivity:
                known_proc = lit_processivity[rt_name]  # nt per binding
                # 50+ nt = highly processive, 5 nt = barely processive
                lit_score = min(known_proc / 50.0, 1.0)
                # Override with weighted combination
                # Literature data is worth more than structural prediction
            else:
                lit_score = None
            
            sub_scores = {
                "thumb_detected": thumb_detected,
                "thumb_quality": thumb_quality,
                "structural_sim": structural_sim,
                "hairpin": hairpin_score,
                "length_proxy": length_score,
            }
            
            if lit_score is not None:
                sub_scores["literature_processivity"] = lit_score
            
            # Aggregate
            log_score = sum(
                self.weights.get(k, 0.1) * np.log(max(v, 1e-6))
                for k, v in sub_scores.items()
            )
            score = float(np.clip(np.exp(log_score), 0, 1))
            
            failure = ""
            if thumb_detected == 0:
                failure = "No thumb domain detected — likely non-processive"
            elif seq_len < self.MIN_PROCESSIVE_LENGTH:
                failure = f"Very short ({seq_len}aa) — insufficient for processivity"
            
            results.append(GateResult(
                rt_name=rt_name, gate_name=self.name, score=score,
                sub_scores=sub_scores,
                confidence=0.8 if thumb_detected else 0.4,
                failure_reason=failure
            ))
        
        return results
Important warning about FoldSeek TM-scores: These are explicitly flagged in the challenge as correlated with family membership. Using foldseek_TM_MMLV in Gate 5 means this gate partly encodes family identity. The key question is whether this is biophysically justified (structural similarity to MMLV genuinely predicts processivity) or confounded (it just identifies retroviral RTs). For LOFO, this feature will help on non-retroviral folds but may artificially boost retroviral predictions. Monitor this gate's contribution to retroviral fold predictions specifically.

5. Utility Functions
pythonCopy# utils/geometry.py

import numpy as np
from scipy.spatial import cKDTree

def compute_steric_clashes(
    query_coords: np.ndarray,    # (N, 3)
    env_coords: np.ndarray,      # (M, 3) 
    clash_dist: float = 2.0,
    contact_dist: float = 4.0,
) -> tuple[int, int]:
    """
    Count atoms in query that clash with environment.
    
    Returns:
        n_clashes: atoms closer than clash_dist
        n_contacts: atoms closer than contact_dist
    """
    tree = cKDTree(env_coords)
    
    n_clashes = 0
    n_contacts = 0
    
    for coord in query_coords:
        neighbors = tree.query_ball_point(coord, contact_dist)
        n_contacts += len(neighbors)
        
        close_neighbors = tree.query_ball_point(coord, clash_dist)
        n_clashes += len(close_neighbors)
    
    return n_clashes, n_contacts


def get_terminus_flexibility(pdb_path: str, n_residues: int = 10) -> float:
    """
    Mean pLDDT of first n_residues (N-terminus).
    Low pLDDT = flexible/disordered = good for linker attachment.
    Returns pLDDT/100 (0-1).
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rt", pdb_path)
    
    b_factors = []
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain.get_residues()):
                if i >= n_residues:
                    break
                for atom in residue:
                    if atom.name == "CA":
                        b_factors.append(atom.get_bfactor())
    
    return np.mean(b_factors) / 100.0 if b_factors else 0.5
pythonCopy# utils/alignment.py

import subprocess
import tempfile
import os
import numpy as np
from dataclasses import dataclass

@dataclass
class AlignmentResult:
    tm_score: float
    rmsd: float
    rotation: np.ndarray          # (3, 3)
    translation: np.ndarray       # (3,)
    aligned_residue_pairs: list   # [(mobile_idx, target_idx), ...]
    aligned_atoms: np.ndarray     # Transformed coordinates (N, 3)
    active_site_center_transformed: np.ndarray  # (3,) or None

def structural_align(
    mobile_pdb: str,
    target_pdb: str = None,
    target_atoms: list = None,
    method: str = "tmalign"
) -> AlignmentResult:
    """
    Structural alignment using TM-align.
    
    Install: conda install -c bioconda tmalign
    Or use tmtools Python package: pip install tmtools
    """
    if method == "tmalign":
        return _tmalign(mobile_pdb, target_pdb)
    elif method == "biopython":
        return _biopython_align(mobile_pdb, target_atoms)
    else:
        raise ValueError(f"Unknown method: {method}")

def _tmalign(mobile_pdb: str, target_pdb: str) -> AlignmentResult:
    """Run TM-align and parse output."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = os.path.join(tmpdir, "aligned")
        
        cmd = [
            "TMalign",
            mobile_pdb,
            target_pdb,
            "-o", out_prefix
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        # Parse TM-score (normalised by target length)
        tm_score = 0.0
        rmsd = 999.0
        
        for line in output.split("\n"):
            if "TM-score=" in line and "Chain_2" in line:
                # TM-score normalised by target (reference)
                tm_score = float(line.split("TM-score=")[1].split()[0])
            if "RMSD" in line and "=" in line:
                try:
                    rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
                except:
                    pass
        
        # Parse rotation matrix from output file
        rotation = np.eye(3)
        translation = np.zeros(3)
        
        matrix_file = out_prefix + "_matrix.txt"
        if os.path.exists(matrix_file):
            rotation, translation = _parse_tmalign_matrix(matrix_file)
    
    return AlignmentResult(
        tm_score=tm_score,
        rmsd=rmsd,
        rotation=rotation,
        translation=translation,
        aligned_residue_pairs=[],
        aligned_atoms=np.array([]),
        active_site_center_transformed=None
    )

def _parse_tmalign_matrix(matrix_file: str):
    """Parse the rotation matrix from TM-align output."""
    rotation = np.eye(3)
    translation = np.zeros(3)
    
    with open(matrix_file) as f:
        lines = f.readlines()
    
    # TM-align matrix format:
    #  ------ The rotation matrix to rotate Chain_1 to Chain_2 ------
    #  m          t(m)        u(m,1)        u(m,2)        u(m,3)
    #  1     tx             r11           r12           r13
    #  2     ty             r21           r22           r23
    #  3     tz             r31           r32           r33
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 5 and parts[0] in ["1", "2", "3"]:
            idx = int(parts[0]) - 1
            translation[idx] = float(parts[1])
            rotation[idx] = [float(parts[2]), float(parts[3]), float(parts[4])]
    
    return rotation, translation

6. Calibration Layer
This is where the 57 samples are used — not to train the gates, but to learn how gate scores combine into a final prediction.
pythonCopy# calibration/integrator.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional

class GateIntegrator:
    """
    Combines gate scores into final predictions.
    
    Design philosophy:
    - Gate scores are computed WITHOUT labels (external knowledge only)
    - This integrator is the ONLY component that sees activity labels
    - It learns minimal parameters: how to weight/threshold gate scores
    - Three integration strategies of increasing complexity:
    
    Strategy A: Multiplicative (no parameters learned from data)
        P(active) = product of gate scores
        Threshold chosen to maximise F1 on training fold
    
    Strategy B: Bayesian Logistic Regression (5-15 parameters)
        logit(P(active)) = β₀ + Σ βᵢ · gate_score_i
        Horseshoe prior on β for automatic relevance determination
        
    Strategy C: BART (many trees, Bayesian regularisation)
        Flexible nonlinear combination of gate scores
        Captures interactions (e.g., an RT needs BOTH good catalysis 
        AND good fusion compatibility)
    """
    
    def __init__(self, strategy: str = "bayesian_lr"):
        self.strategy = strategy
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
    
    def _prepare_features(
        self,
        gate_scores: pd.DataFrame,
        handcrafted_residuals: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Construct feature matrix from gate scores.
        
        gate_scores: DataFrame with columns like 
            gate1_score, gate1_sub1, ..., gate2_score, ...
        handcrafted_residuals: Optional additional features
            (handcrafted features not captured by gates)
        """
        # Primary features: gate-level scores
        gate_cols = [c for c in gate_scores.columns if c.endswith("_score")]
        X = gate_scores[gate_cols].values
        
        # Optional: add a few handcrafted features as residual signal
        if handcrafted_residuals is not None:
            X = np.hstack([X, handcrafted_residuals.values])
        
        return X
    
    def fit_predict_lofo(
        self,
        gate_scores: pd.DataFrame,
        labels: pd.Series,
        families: pd.Series,
        handcrafted_residuals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Leave-one-family-out cross-validation.
        
        For each family:
          1. Hold out that family
          2. Fit integrator on remaining families
          3. Predict held-out family
          4. Record predictions
        
        Returns DataFrame with predictions for all 57 RTs.
        """
        X = self._prepare_features(gate_scores, handcrafted_residuals)
        y = labels.values
        family_arr = families.values
        
        unique_families = np.unique(family_arr)
        
        all_predictions = []
        
        for held_out_family in unique_families:
            test_mask = family_arr == held_out_family
            train_mask = ~test_mask
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]
            
            # Fit scaler on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            if self.strategy == "multiplicative":
                scores = self._multiplicative(X_test)
            elif self.strategy == "bayesian_lr":
                scores = self._bayesian_lr_fit_predict(
                    X_train_scaled, y_train, X_test_scaled
                )
            elif self.strategy == "bart":
                scores = self._bart_fit_predict(
                    X_train_scaled, y_train, X_test_scaled
                )
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Optimise threshold on training set
            train_scores = self._get_train_scores(
                X_train_scaled, y_train, self.strategy
            )
            threshold = self._optimise_threshold(train_scores, y_train)
            
            # Record predictions
            test_names = gate_scores.index[test_mask]
            for i, (name, score) in enumerate(zip(test_names, scores)):
                all_predictions.append({
                    "rt_name": name,
                    "held_out_family": held_out_family,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                })
        
        return pd.DataFrame(all_predictions)
    
    def _multiplicative(self, X: np.ndarray) -> np.ndarray:
        """
        No parameters. Just multiply gate scores.
        Each column of X is a gate score ∈ [0, 1].
        """
        # Geometric mean across gates
        log_scores = np.log(np.clip(X, 1e-6, 1.0))
        return np.exp(np.mean(log_scores, axis=1))
    
    def _bayesian_lr_fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Bayesian logistic regression with horseshoe prior.
        Uses PyMC for inference.
        """
        import pymc as pm
        import pytensor.tensor as pt
        
        n_features = X_train.shape[1]
        
        with pm.Model() as model:
            # Horseshoe prior for sparsity
            tau = pm.HalfCauchy("tau", beta=1.0)
            lambdas = pm.HalfCauchy("lambdas", beta=1.0, shape=n_features)
            
            beta = pm.Normal(
                "beta", mu=0, sigma=tau * lambdas, shape=n_features
            )
            intercept = pm.Normal("intercept", mu=0, sigma=2.0)
            
            # Class weight adjustment
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            weights = np.where(y_train == 1, n_neg / len(y_train), n_pos / len(y_train))
            
            logits = intercept + pt.dot(X_train, beta)
            
            pm.Bernoulli("obs", logit_p=logits, observed=y_train, 
                        shape=len(y_train))
            
            # Sample
            trace = pm.sample(
                2000, tune=1000, cores=2,
                target_accept=0.9, random_seed=42,
                progressbar=False
            )
        
        # Predict on test set
        beta_post = trace.posterior["beta"].values.reshape(-1, n_features)
        intercept_post = trace.posterior["intercept"].values.flatten()
        
        logits_test = X_test @ beta_post.T + intercept_post[np.newaxis, :]
        probs_test = 1.0 / (1.0 + np.exp(-logits_test))
        
        # Mean posterior predictive probability
        return probs_test.mean(axis=1)
    
    def _bart_fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Bayesian Additive Regression Trees.
        Uses PyMC-BART.
        """
        import pymc as pm
        import pymc_bart as pmb
        
        with pm.Model() as model:
            # BART prior
            mu = pmb.BART(
                "mu", X=X_train, Y=y_train,
                m=50,  # number of trees (conservative for small n)
            )
            
            p = pm.Deterministic("p", pm.math.sigmoid(mu))
            
            pm.Bernoulli("obs", p=p, observed=y_train)
            
            trace = pm.sample(
                1000, tune=500, cores=2,
                random_seed=42, progressbar=False
            )
        
        # Predict
        with model:
            pm.set_data({"X": X_test})  # This needs BART-specific prediction
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
        
        return ppc.posterior_predictive["obs"].mean(dim=["chain", "draw"]).values
    
    def _optimise_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Find threshold that maximises F1 on training data."""
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.02):
            preds = (scores >= thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        return best_thresh

7. Evaluation Module
pythonCopy# calibration/evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score
)
from scipy.stats import spearmanr, kendalltau

def evaluate_lofo_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Full LOFO evaluation matching challenge requirements.
    
    predictions: from GateIntegrator.fit_predict_lofo()
    ground_truth: rt_sequences.csv with 'active' and 'pe_efficiency_pct'
    """
    # Merge
    merged = predictions.merge(
        ground_truth[["rt_name", "active", "pe_efficiency_pct", "rt_family"]],
        on="rt_name"
    )
    
    y_true = merged["active"].values
    y_pred = merged["predicted_active"].values
    y_score = merged["predicted_score"].values
    
    # ---- Overall LOFO metrics ----
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC only if both classes present
    if len(np.unique(y_true)) > 1:
        overall_auc = roc_auc_score(y_true, y_score)
    else:
        overall_auc = None
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # ---- Per-family breakdown ----
    family_results = {}
    informative_f1s = []
    
    INFORMATIVE_FAMILIES = [
        "Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"
    ]
    
    for family in merged["held_out_family"].unique():
        fam_mask = merged["held_out_family"] == family
        fam_true = merged.loc[fam_mask, "active"].values
        fam_pred = merged.loc[fam_mask, "predicted_active"].values
        fam_score = merged.loc[fam_mask, "predicted_score"].values
        
        fam_tp = ((fam_true == 1) & (fam_pred == 1)).sum()
        fam_fp = ((fam_true == 0) & (fam_pred == 1)).sum()
        fam_fn = ((fam_true == 1) & (fam_pred == 0)).sum()
        fam_tn = ((fam_true == 0) & (fam_pred == 0)).sum()
        fam_n_active = fam_true.sum()
        
        fam_f1 = f1_score(fam_true, fam_pred, zero_division=0) \
                 if len(np.unique(fam_true)) > 1 else None
        
        family_results[family] = {
            "n": len(fam_true),
            "n_active": int(fam_n_active),
            "tp": int(fam_tp),
            "fp": int(fam_fp),
            "fn": int(fam_fn),
            "tn": int(fam_tn),
            "f1": fam_f1,
            "tp_rate": f"{fam_tp}/{fam_n_active}" if fam_n_active > 0 else "N/A",
        }
        
        if family in INFORMATIVE_FAMILIES and fam_f1 is not None:
            informative_f1s.append(fam_f1)
    
    # ---- Primary metric: Macro F1 across 4 informative folds ----
    lofo_macro_f1 = np.mean(informative_f1s) if informative_f1s else 0.0
    
    # ---- Retroviral fold specifically ----
    retro = family_results.get("Retroviral", {})
    retroviral_tp_12 = retro.get("tp", 0)
    
    # ---- Ranking quality (active RTs only) ----
    active_mask = merged["active"] == 1
    if active_mask.sum() > 2:
        active_efficiency = merged.loc[active_mask, "pe_efficiency_pct"].values
        active_scores = merged.loc[active_mask, "predicted_score"].values
        
        spearman_rho, spearman_p = spearmanr(active_efficiency, active_scores)
        kendall_tau, kendall_p = kendalltau(active_efficiency, active_scores)
    else:
        spearman_rho = kendall_tau = None
    
    # ---- Compile results ----
    results = {
        "primary_metric": {
            "LOFO_macro_F1_4_folds": float(lofo_macro_f1),
        },
        "secondary_metrics": {
            "retroviral_TP_of_12": retroviral_tp_12,
            "overall_F1": float(overall_f1),
            "overall_AUC": float(overall_auc) if overall_auc else None,
            "total_TP": int(tp),
            "total_FP": int(fp),
            "total_FN": int(fn),
            "total_TN": int(tn),
        },
        "ranking_quality": {
            "spearman_rho": float(spearman_rho) if spearman_rho else None,
            "kendall_tau": float(kendall_tau) if kendall_tau else None,
        },
        "per_family": family_results,
    }
    
    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    
    print("=" * 65)
    print("  LOFO EVALUATION RESULTS")
    print("=" * 65)
    
    pm = results["primary_metric"]
    print(f"\n  PRIMARY: LOFO Macro-F1 (4 informative folds): "
          f"{pm['LOFO_macro_F1_4_folds']:.3f}")
    
    sm = results["secondary_metrics"]
    print(f"\n  RETROVIRAL WALL: {sm['retroviral_TP_of_12']}/12 TPs")
    print(f"  Overall F1: {sm['overall_F1']:.3f}")
    print(f"  Overall AUC: {sm['overall_AUC']:.3f}" if sm['overall_AUC'] else "")
    print(f"  Confusion: TP={sm['total_TP']} FP={sm['total_FP']} "
          f"FN={sm['total_FN']} TN={sm['total_TN']}")
    
    rq = results["ranking_quality"]
    if rq["spearman_rho"]:
        print(f"\n  Ranking: Spearman ρ={rq['spearman_rho']:.3f}, "
              f"Kendall τ={rq['kendall_tau']:.3f}")
    
    print(f"\n  Per-Family Breakdown:")
    print(f"  {'Family':<25} {'n':>3} {'Active':>6} {'TP':>4} {'FP':>4} "
          f"{'F1':>6} {'TP Rate':>8}")
    print(f"  {'-'*60}")
    
    for fam, fr in results["per_family"].items():
        f1_str = f"{fr['f1']:.3f}" if fr['f1'] is not None else "N/A"
        print(f"  {fam:<25} {fr['n']:>3} {fr['n_active']:>6} "
              f"{fr['tp']:>4} {fr['fp']:>4} {f1_str:>6} {fr['tp_rate']:>8}")
    
    print("=" * 65)

8. Main Pipeline Orchestrator
pythonCopy# main.py

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from config import DATA_DIR, EXTERNAL_DIR, OUTPUT_DIR
from download_external import download_all_external
from gates.gate1_foldability import FoldabilityGate
from gates.gate2_fusion import FusionCompatibilityGate
from gates.gate3_substrate_binding import SubstrateBindingGate
from gates.gate4_catalytic import CatalyticCompetenceGate
from gates.gate5_processivity import ProcessivityGate
from calibration.integrator import GateIntegrator
from calibration.evaluation import evaluate_lofo_predictions, print_results
from analysis.gate_diagnostics import diagnose_gates

def main():
    """Full pipeline: data → gates → calibration → evaluation → output."""
    
    print("=" * 65)
    print("  RETROVIRAL WALL CHALLENGE - MECHANISTIC GATE PIPELINE")
    print(f"  Run: {datetime.now().isoformat()}")
    print("=" * 65)
    
    # ================================================================
    # STAGE 0: Load data & external resources
    # ================================================================
    print("\n[Stage 0] Loading data...")
    
    sequences = pd.read_csv(f"{DATA_DIR}/rt_sequences.csv")
    handcrafted = pd.read_csv(f"{DATA_DIR}/handcrafted_features.csv")
    families = pd.read_csv(f"{DATA_DIR}/family_splits.csv")
    
    # Download external PDBs if needed
    print("[Stage 0] Checking external data...")
    external_data = download_all_external()
    
    print(f"  Loaded {len(sequences)} RTs, {handcrafted.shape[1]} features")
    print(f"  Active: {sequences['active'].sum()}, "
          f"Inactive: {(~sequences['active'].astype(bool)).sum()}")
    
    # ================================================================
    # STAGE 1: Compute gate scores (NO LABELS USED)
    # ================================================================
    print("\n[Stage 1] Computing gate scores...")
    
    gates = [
        FoldabilityGate(),
        FusionCompatibilityGate(),
        SubstrateBindingGate(),
        CatalyticCompetenceGate(),
        ProcessivityGate(),
    ]
    
    all_gate_scores = pd.DataFrame({"rt_name": sequences["rt_name"]})
    all_gate_scores = all_gate_scores.set_index("rt_name")
    
    for gate in gates:
        print(f"  Computing {gate.name}...")
        results = gate.compute_scores(
            sequences=sequences,
            structures_dir=f"{DATA_DIR}/structures",
            handcrafted=handcrafted,
            external_data=external_data,
        )
        
        gate_df = gate.score_matrix(results).set_index("rt_name")
        all_gate_scores = all_gate_scores.join(gate_df)
        
        # Quick diagnostic: correlation with activity
        scores = gate_df[f"{gate.name}_score"]
        active = sequences.set_index("rt_name")["active"]
        corr = scores.corr(active)
        print(f"    {gate.name} score ↔ activity correlation: {corr:.3f}")
    
    # Save gate scores
    gate_scores_path = f"{OUTPUT_DIR}/gate_scores/all_gate_scores.csv"
    os.makedirs(os.path.dirname(gate_scores_path), exist_ok=True)
    all_gate_scores.to_csv(gate_scores_path)
    print(f"  Saved gate scores to {gate_scores_path}")
    
    # ================================================================
    # STAGE 2: Gate diagnostics (before calibration)
    # ================================================================
    print("\n[Stage 2] Gate diagnostics...")
    
    diagnostics = diagnose_gates(
        gate_scores=all_gate_scores,
        labels=sequences.set_index("rt_name")["active"],
        families=sequences.set_index("rt_name")["rt_family"],
    )
    # This checks:
    # - Per-gate discriminative power (AUC per gate)
    # - Per-gate correlation with family (confounding check)
    # - Which gates separate active from inactive WITHIN families
    # - Whether any single gate breaks the retroviral wall
    
    # ================================================================
    # STAGE 3: Calibration with LOFO
    # ================================================================
    print("\n[Stage 3] Calibration & LOFO evaluation...")
    
    strategies = ["multiplicative", "bayesian_lr", "bart"]
    best_result = None
    best_strategy = None
    best_macro_f1 = -1
    
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        
        integrator = GateIntegrator(strategy=strategy)
        predictions = integrator.fit_predict_lofo(
            gate_scores=all_gate_scores,
            labels=sequences.set_index("rt_name")["active"],
            families=sequences.set_index("rt_name")["rt_family"],
            handcrafted_residuals=None,  # Start with gates only
        )
        
        results = evaluate_lofo_predictions(
            predictions=predictions,
            ground_truth=sequences,
        )
        
        print_results(results)
        
        macro_f1 = results["primary_metric"]["LOFO_macro_F1_4_folds"]
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_result = results
            best_strategy = strategy
            best_predictions = predictions
    
    # ================================================================
    # STAGE 4: Add handcrafted residuals (optional boost)
    # ================================================================
    print(f"\n[Stage 4] Testing with handcrafted residuals...")
    
    # Select handcrafted features NOT already captured by gates
    # and with low correlation to family
    residual_features = select_residual_features(
        handcrafted=handcrafted,
        gate_scores=all_gate_scores,
        families=sequences["rt_family"],
        max_features=5,           # Keep it minimal
        max_family_corr=0.5,      # Low family correlation
    )
    
    if residual_features is not None:
        integrator = GateIntegrator(strategy=best_strategy)
        predictions_augmented = integrator.fit_predict_lofo(
            gate_scores=all_gate_scores,
            labels=sequences.set_index("rt_name")["active"],
            families=sequences.set_index("rt_name")["rt_family"],
            handcrafted_residuals=residual_features,
        )
        
        results_augmented = evaluate_lofo_predictions(
            predictions_augmented, sequences
        )
        
        aug_f1 = results_augmented["primary_metric"]["LOFO_macro_F1_4_folds"]
        if aug_f1 > best_macro_f1:
            print(f"  Handcrafted residuals improved F1: "
                  f"{best_macro_f1:.3f} → {aug_f1:.3f}")
            best_result = results_augmented
            best_predictions = predictions_augmented
            best_macro_f1 = aug_f1
    
    # ================================================================
    # STAGE 5: Generate submission
    # ================================================================
    print(f"\n[Stage 5] Generating submission...")
    
    submission = best_predictions[["rt_name", "predicted_active", "predicted_score"]]
    submission_path = f"{OUTPUT_DIR}/predictions/submission.csv"
    submission.to_csv(submission_path, index=False)
    
    # Save full results
    results_path = f"{OUTPUT_DIR}/predictions/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(best_result, f, indent=2, default=str)
    
    print(f"  Submission: {submission_path}")
    print(f"  Results: {results_path}")
    print(f"\n  BEST STRATEGY: {best_strategy}")
    print(f"  LOFO Macro-F1: {best_macro_f1:.3f}")
    print(f"  Retroviral TP: {best_result['secondary_metrics']['retroviral_TP_of_12']}/12")
    
    return best_result


def select_residual_features(
    handcrafted, gate_scores, families, max_features=5, max_family_corr=0.5
):
    """
    Select handcrafted features that:
    1. Are NOT highly correlated with any gate score (residual information)
    2. Are NOT highly correlated with family identity (family-agnostic)
    3. Have non-trivial variance
    """
    from sklearn.preprocessing import LabelEncoder
    
    hc = handcrafted.set_index("rt_name")
    
    # Encode family as numeric for correlation
    le = LabelEncoder()
    family_encoded = le.fit_transform(families)
    
    selected = []
    
    for col in hc.columns:
        values = hc[col].values
        
        # Skip if too many NaN
        if np.isnan(values).mean() > 0.3:
            continue
        
        # Fill NaN with median for correlation computation
        values_filled = np.where(np.isnan(values), np.nanmedian(values), values)
        
        # Skip if no variance
        if np.std(values_filled) < 1e-6:
            continue
        
        # Check family correlation (Cramér's V or eta-squared)
        from scipy.stats import f_oneway
        family_groups = [values_filled[family_encoded == f] 
                        for f in np.unique(family_encoded)]
        family_groups = [g for g in family_groups if len(g) > 1]
        
        if len(family_groups) > 1:
            f_stat, p_val = f_oneway(*family_groups)
            # Eta-squared
            ss_between = sum(len(g) * (g.mean() - values_filled.mean())**2 
                           for g in family_groups)
            ss_total = np.sum((values_filled - values_filled.mean())**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 1.0
            
            if eta_sq > max_family_corr:
                continue  # Too correlated with family
        
        # Check correlation with gate scores
        max_gate_corr = max(
            abs(np.corrcoef(values_filled, gate_scores[col2].fillna(0).values)[0, 1])
            for col2 in gate_scores.columns if col2.endswith("_score")
        )
        
        if max_gate_corr > 0.7:
            continue  # Already captured by gates
        
        selected.append(col)
    
    if len(selected) == 0:
        return None
    
    # Rank by univariate discrimination (Mann-Whitney U)
    from scipy.stats import mannwhitneyu
    active = handcrafted.set_index("rt_name").loc[
        gate_scores.index
    ].index  # This needs the label - OK because it's for feature selection
    # ... (compute U statistic for each selected feature)
    
    # Return top max_features
    return hc[selected[:max_features]]


if __name__ == "__main__":
    results = main()

9. Gate Diagnostic Module
pythonCopy# analysis/gate_diagnostics.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, f_oneway
import matplotlib.pyplot as plt

def diagnose_gates(
    gate_scores: pd.DataFrame,
    labels: pd.Series,
    families: pd.Series,
) -> dict:
    """
    Comprehensive diagnostic of each gate's discriminative power.
    
    For each gate score, compute:
    1. Overall AUC (active vs inactive)
    2. Per-family AUC (within-family discrimination)
    3. Family confounding (how much does this score correlate with family?)
    4. Retroviral wall test (does this score separate retroviral actives 
       from retroviral inactives?)
    5. Cross-family test (trained on non-retroviral, evaluated on retroviral)
    """
    
    score_cols = [c for c in gate_scores.columns if c.endswith("_score")]
    
    diagnostics = {}
    
    for col in score_cols:
        scores = gate_scores[col].values
        y = labels.values
        fam = families.values
        
        # Handle NaN
        valid = ~np.isnan(scores)
        if valid.sum() < 20:
            continue
        
        s, y_v, f_v = scores[valid], y[valid], fam[valid]
        
        # 1. Overall AUC
        if len(np.unique(y_v)) > 1:
            auc = roc_auc_score(y_v, s)
        else:
            auc = None
        
        # 2. Mann-Whitney U (active vs inactive scores)
        active_scores = s[y_v == 1]
        inactive_scores = s[y_v == 0]
        if len(active_scores) > 0 and len(inactive_scores) > 0:
            u_stat, u_pval = mannwhitneyu(active_scores, inactive_scores, 
                                          alternative="greater")
        else:
            u_stat, u_pval = None, None
        
        # 3. Family confounding (eta-squared from one-way ANOVA)
        family_groups = [s[f_v == f] for f in np.unique(f_v)]
        family_groups = [g for g in family_groups if len(g) > 1]
        if len(family_groups) > 1:
            ss_between = sum(len(g) * (g.mean() - s.mean())**2 
                           for g in family_groups)
            ss_total = np.sum((s - s.mean())**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
        else:
            eta_sq = 0
        
        # 4. Within-retroviral discrimination
        retro_mask = f_v == "Retroviral"
        if retro_mask.sum() > 0 and len(np.unique(y_v[retro_mask])) > 1:
            retro_auc = roc_auc_score(y_v[retro_mask], s[retro_mask])
        else:
            retro_auc = None
        
        # 5. Cross-family: score threshold from non-retroviral → retroviral TP
        non_retro = ~retro_mask
        if non_retro.sum() > 5 and len(np.unique(y_v[non_retro])) > 1:
            # Find threshold on non-retroviral that maximises F1
            best_thresh = _best_threshold(s[non_retro], y_v[non_retro])
            # Apply to retroviral
            retro_preds = (s[retro_mask] >= best_thresh).astype(int)
            retro_tp = ((y_v[retro_mask] == 1) & (retro_preds == 1)).sum()
            retro_n_active = (y_v[retro_mask] == 1).sum()
        else:
            retro_tp, retro_n_active = 0, 0
        
        diagnostics[col] = {
            "overall_auc": auc,
            "mann_whitney_p": u_pval,
            "family_eta_squared": eta_sq,
            "retroviral_within_auc": retro_auc,
            "cross_family_retroviral_tp": f"{retro_tp}/{retro_n_active}",
            "active_mean": float(active_scores.mean()),
            "inactive_mean": float(inactive_scores.mean()),
        }
        
        print(f"  {col}:")
        print(f"    AUC={auc:.3f}" if auc else "    AUC=N/A", end="")
        print(f"  Family η²={eta_sq:.3f}", end="")
        print(f"  Retro AUC={retro_auc:.3f}" if retro_auc else "", end="")
        print(f"  Cross-family Retro TP: {retro_tp}/{retro_n_active}")
    
    return diagnostics


def _best_threshold(scores, labels):
    """Find threshold maximising F1."""
    from sklearn.metrics import f1_score
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t

10. Configuration
pythonCopy# config.py

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create output directories
for d in [OUTPUT_DIR, f"{OUTPUT_DIR}/gate_scores", 
          f"{OUTPUT_DIR}/predictions", f"{OUTPUT_DIR}/figures"]:
    os.makedirs(d, exist_ok=True)

# External PDB structures
EXTERNAL_PDBS = {
    "pe2_complex": {
        "pdb_id": "8W8H",
        "description": "Prime editor cryo-EM structure",
        "chains": {"cas9": "A", "mmlv_rt": "B", "target_dna": "C,D"},
        "path": os.path.join(EXTERNAL_DIR, "pdb", "8W8H.pdb"),
    },
    "mmlv_rt_substrate": {
        "pdb_id": "7UVO",
        "description": "MMLV-RT with template-primer",
        "path": os.path.join(EXTERNAL_DIR, "pdb", "7UVO.pdb"),
    },
    "hiv1_rt_substrate": {
        "pdb_id": "1RTD",
        "description": "HIV-1 RT with RNA:DNA hybrid",
        "path": os.path.join(EXTERNAL_DIR, "pdb", "1RTD.pdb"),
    },
}

# Cas9 parameters (SpCas9 from S. pyogenes)
CAS9_SEQUENCE_LENGTH = 1368  # aa
CAS9_NICKASE_MUTATION = "H840A"  # nCas9 for PE

# Prime editor architecture
PE_LINKER_SEQUENCE = "SGGSSGGSSGSETPGTSESATPESSGGSSGGS"  # Standard PE2 linker
PE_FUSION_POINT = "C-terminal"  # RT fused to C-terminus of Cas9

# Gate computation parameters
GATE_PARAMS = {
    "clash_distance_A": 2.0,
    "contact_distance_A": 4.0,
    "groove_contact_distance_A": 6.0,
    "ref_triad_d1_d2_A": 6.5,
    "ref_triad_d2_d3_A": 3.8,
    "triad_distance_tolerance_A": 3.0,
    "min_processive_length_aa": 200,
    "max_size_no_penalty_aa": 800,
}

# Calibration parameters
CALIBRATION = {
    "bayesian_lr_samples": 2000,
    "bayesian_lr_tune": 1000,
    "bart_n_trees": 50,
    "bart_samples": 1000,
    "random_seed": 42,
}

11. Requirements
Copy# requirements.txt

# Core
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3

# Structure analysis
biopython>=1.81
# tmtools  # or install TM-align binary

# Bayesian modelling
pymc>=5.10
pymc-bart>=0.5
arviz>=0.17

# Visualisation
matplotlib>=3.7
seaborn>=0.12

# Optional: electrostatics
# pdb2pqr  # for APBS electrostatics
# freesasa  # for SASA calculations

# Optional: docking
# hdock  # for protein-nucleic acid docking

12. Implementation Timeline
CopyWeek 1: Foundation
├── Day 1-2: Set up project structure, load all data, EDA notebook
│            Inspect provided PDB structures (pLDDT distributions)
│            Identify exact column names in handcrafted_features.csv
│            Download and inspect external PDB structures
│
├── Day 3-4: Implement Gate 1 (foldability) and Gate 4 (catalytic)
│            These are the easiest — mostly feature extraction
│            Run gate diagnostics on these two gates
│
├── Day 5-7: Implement Gate 5 (processivity)
│            Curate literature processivity data
│            Run diagnostics

Week 2: The Hard Gates
├── Day 8-10: Implement Gate 2 (fusion compatibility)
│             This is the novel contribution
│             Start with structural superposition approach
│             Inspect PE2 cryo-EM structure carefully
│             Identify chain boundaries and coordinate systems
│
├── Day 11-12: Implement Gate 3 (substrate binding groove)
│              Groove identification via structural alignment
│              Electrostatic analysis
│
├── Day 13-14: Full gate diagnostics
│              Which gates discriminate? Which are confounded?
│              Identify the retroviral wall breakers

Week 3: Calibration & Iteration
├── Day 15-16: Implement calibration layer
│              Test all three strategies (multiplicative, BLR, BART)
│              LOFO evaluation
│
├── Day 17-18: Iterate on weak gates
│              If Gate 2 doesn't discriminate, investigate why
│              Try alternative approaches for the weakest gate
│
├── Day 19-21: Add handcrafted residuals if they help
│              Ablation studies (which gates matter most?)
│              Per-residue PLM embeddings at catalytic sites (if time)

Week 4: Polish & Submit
├── Day 22-23: Final evaluation, threshold optimisation
├── Day 24-25: Write 2-page report
├── Day 26:    Clean code, ensure reproducibility
├── Day 27:    Submit

13. Sanity Checks (implement as unit tests)
pythonCopy# tests/test_known_rts.py

"""
Sanity checks that known RTs get sensible gate scores.
These are NOT using labels for training — they verify that
the gates produce biologically sensible outputs.
"""

def test_mmlv_passes_all_gates():
    """MMLV-RT is the gold standard (41% PE efficiency). 
    It MUST score high on all gates."""
    scores = get_gate_scores("MMLV-RT")
    assert scores["foldability_score"] > 0.6
    assert scores["fusion_compat_score"] > 0.7  # It's what the fusion was designed for
    assert scores["substrate_binding_score"] > 0.6
    assert scores["catalytic_score"] > 0.7
    assert scores["processivity_score"] > 0.7

def test_inactive_rt_fails_at_least_one_gate():
    """At least one gate should be low for known inactive RTs."""
    for rt_name in KNOWN_INACTIVE_RTS:
        scores = get_gate_scores(rt_name)
        min_score = min(scores.values())
        assert min_score < 0.5, f"{rt_name} passes all gates but is inactive"

def test_gates_independent_of_labels():
    """Verify that gate computation doesn't access activity labels."""
    # Run gates with labels shuffled — scores should be identical
    # (since gates don't use labels)
    scores_real = compute_all_gates(labels=real_labels)
    scores_shuffled = compute_all_gates(labels=shuffled_labels)
    assert_frame_equal(scores_real, scores_shuffled)

def test_gate_scores_bounded():
    """All gate scores should be in [0, 1]."""
    for col in gate_score_columns:
        assert gate_scores[col].min() >= 0
        assert gate_scores[col].max() <= 1

14. Key Risk Mitigations
RiskMitigationPE2 cryo-EM structure doesn't cleanly separate chainsManually inspect PDB, use RCSB viewer, identify chains before codingTM-align fails for very divergent RTs (TM < 0.2)Default to size-based and sequence-based proxies for those RTsGate 2 doesn't discriminate (all RTs score similarly)Fall back to simpler fusion proxies: RT size + terminal pLDDT + surface charge near N-termBayesian models don't converge with 5-6 features and ~40 training samplesUse variational inference (faster) or fall back to regularised logistic regressionAll gates correlate with family identityReport family η² for each gate; accept that some family signal is biophysically real; rely on the subset of gate variance that is family-independentCompounding gate errors give noisy final scoresThe calibration layer's job is to handle this; BART can