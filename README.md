# ETAF-Agent

ETAF-Agent contains research components for evidence-tier-aware drug ranking in uveal melanoma, including genomic representation, evidence aggregation, pathway scoring, ensemble voting, and hash commitments.

## Installation

```bash
pip install -e ".[dev]"
```

## Data and execution

Patient cohorts, drug records, pathway graphs, evidence records, independent evaluation references, and pretrained model weights are not included. Dataset readers, model providers, and evidence integrations must be implemented before training, ranking, or evaluation. The command-line workflow currently stops with an error when it attempts to load these inputs.

The genomic tower is a trainable implementation and does not include pretrained scPRINT weights. MolFormer and ranking model providers are not integrated. Reported study results have not been reproduced by this repository.

Components are organized under `etaf_agent/`: `vestibule` defines data structures, `tumbler` handles genomic encoding, `ward` provides evidence aggregation, `linkage` handles pathway interfaces, `bolt` provides scoring, `quorum` provides voting, `seal` provides commitments, and `gauge` provides evaluation utilities. Configuration files are in `keyplans/`.

Potential sources include [TCGA-UVM](https://portal.gdc.cancer.gov/projects/TCGA-UVM), [AACR GENIE](https://www.synapse.org/genie), [DepMap](https://depmap.org), [OncoKB](https://www.oncokb.org), [CIViC](https://civicdb.org), and [Reactome](https://reactome.org). Their access and usage terms apply.

```bash
make test
```

This is research software and has not been validated for clinical use.
