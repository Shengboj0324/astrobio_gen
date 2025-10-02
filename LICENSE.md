# LICENSE

## Astrobiology Research Platform: Multi-Modal Deep Learning System for Exoplanet Habitability Assessment

**Copyright © 2025 Astrobiology Platform Research Consortium**

---

## PREAMBLE

This document establishes the comprehensive licensing framework for the Astrobiology Research Platform, a state-of-the-art computational system integrating multi-modal deep learning architectures (13.14 billion parameters), physics-informed neural networks, and graph-based metabolic modeling for scientific investigation of exoplanetary habitability. The platform represents a synthesis of advanced artificial intelligence methodologies, astrophysical simulation frameworks, and biochemical pathway analysis tools designed to advance humanity's understanding of potentially habitable worlds beyond our solar system.

The licensing structure herein reflects the dual imperatives of (1) fostering open scientific collaboration and reproducible research in accordance with FAIR principles (Findable, Accessible, Interoperable, Reusable), and (2) ensuring responsible stewardship of computational resources, data provenance, and intellectual property rights across diverse stakeholder communities including academic institutions, governmental research agencies, and private sector entities.

---

## I. SCOPE AND APPLICABILITY

### 1.1 Covered Works

This license governs all software, documentation, data products, trained model weights, and associated materials contained within the Astrobiology Platform repository, including but not limited to:

- **Computational Models**: Neural network architectures, training algorithms, inference pipelines
- **Scientific Data Processing**: Data acquisition systems, preprocessing pipelines, quality assurance frameworks
- **Research Infrastructure**: Training orchestration, distributed computing integration, monitoring systems
- **Documentation**: Technical specifications, user guides, API references, research protocols
- **Auxiliary Components**: Configuration files, deployment scripts, validation frameworks

### 1.2 Licensing Philosophy

The platform employs a **component-based multi-license framework** to optimize for:

1. **Scientific Openness**: Maximizing accessibility for academic research and educational purposes
2. **Commercial Viability**: Enabling enterprise adoption while preserving attribution requirements
3. **Data Integrity**: Protecting scientific data provenance and quality assurance mechanisms
4. **Privacy Protection**: Ensuring privacy-preserving technologies remain open and auditable
5. **Legal Clarity**: Providing unambiguous terms for diverse use cases and jurisdictions

---

## II. PRIMARY LICENSE: APACHE LICENSE 2.0

### 2.1 Core Research Software Components

**Applicable Components:**

- `/models/` — Neural network architectures including:
  - `rebuilt_llm_integration.py` — 13.14B parameter transformer with Flash Attention 2.0, RoPE, GQA
  - `rebuilt_graph_vae.py` — Graph Transformer Variational Autoencoder (~1.2B parameters)
  - `rebuilt_datacube_cnn.py` — Hybrid CNN-Vision Transformer for 5D climate datacubes (~2.5B parameters)
  - `rebuilt_multimodal_integration.py` — Cross-attention fusion system
  - Advanced research modules (diffusion models, causal AI, autonomous agents)

- `/training/` — Training infrastructure:
  - `unified_sota_training_system.py` — Orchestration framework
  - `enhanced_training_orchestrator.py` — Multi-GPU distributed training
  - `sota_training_strategies.py` — Optimization algorithms

- `/utils/` — System utilities and diagnostics
- `/validation/` — Validation frameworks and benchmarking suites
- `/monitoring/` — Real-time performance monitoring
- `train_unified_sota.py` — Primary training entry point

### 2.2 License Terms

The aforementioned components are licensed under the **Apache License, Version 2.0** (January 2004), the full text of which is available at:

**https://www.apache.org/licenses/LICENSE-2.0**

#### Key Provisions:

1. **Grant of Copyright License** (§2): Perpetual, worldwide, non-exclusive, royalty-free license to reproduce, prepare derivative works, publicly display, publicly perform, sublicense, and distribute the Work and Derivative Works.

2. **Grant of Patent License** (§3): Perpetual, worldwide, non-exclusive, royalty-free patent license to make, use, sell, offer for sale, import, and otherwise transfer the Work.

3. **Redistribution Requirements** (§4):
   - Retain copyright, patent, trademark, and attribution notices
   - Include copy of Apache License 2.0
   - Provide notice of modifications to original files
   - Include NOTICE file if provided with distribution

4. **Submission of Contributions** (§5): Contributions submitted for inclusion are licensed under Apache 2.0 unless explicitly stated otherwise.

5. **Trademarks** (§6): No trademark rights granted except as required for reasonable and customary use in describing origin of Work.

6. **Disclaimer of Warranty** (§7): Work provided "AS IS" without warranties of any kind.

7. **Limitation of Liability** (§8): No liability for damages arising from use of Work.

### 2.3 Rationale for Apache 2.0 Selection

The Apache License 2.0 was selected for core research software based on the following considerations:

- **Patent Protection**: Explicit patent grant (§3) protects users from patent litigation by contributors, critical for enterprise adoption of AI/ML systems where patent landscapes are complex.

- **Enterprise Compatibility**: Permissive terms enable integration into proprietary systems while maintaining attribution requirements, facilitating industry-academic partnerships.

- **Scientific Reproducibility**: Redistribution requirements (§4) ensure proper attribution and modification tracking, supporting scientific reproducibility standards.

- **International Recognition**: Apache 2.0 is OSI-approved, FSF-compatible, and widely recognized across jurisdictions, reducing legal ambiguity for international collaborations.

- **Community Standards**: Adopted by major AI/ML frameworks (TensorFlow, PyTorch ecosystem, Hugging Face Transformers), ensuring ecosystem compatibility.

---

## III. SCIENTIFIC DATA PROCESSING: MIT LICENSE

### 3.1 Applicable Components

**Data Acquisition and Processing Systems:**

- `/data_build/` — Comprehensive data integration framework:
  - `comprehensive_13_sources_integration.py` — Multi-source scientific data acquisition
  - `real_data_storage.py` — Production data management system
  - `unified_dataloader_fixed.py` — PyTorch-compatible data loading
  - `advanced_quality_system.py` — Data quality assurance and validation
  - `metadata_annotation_system.py` — Provenance tracking and metadata management

- `/datamodules/` — PyTorch Lightning data modules for training pipelines
- `/scripts/` — Auxiliary data processing and validation scripts
- `run_comprehensive_data_system.py` — System-wide data orchestration

### 3.2 License Terms

The MIT License (Massachusetts Institute of Technology License) is a permissive free software license originating from MIT. Full text available at:

**https://opensource.org/licenses/MIT**

```
MIT License

Copyright © 2025 Astrobiology Platform Research Consortium

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.3 Rationale for MIT License Selection

- **Maximal Permissiveness**: Minimal restrictions facilitate integration with diverse research computing environments and proprietary institutional systems.

- **Academic Standard**: MIT License is the de facto standard for scientific software in computational biology, astrophysics, and data science communities.

- **Compatibility**: Compatible with Apache 2.0, BSD, and proprietary licenses, enabling seamless integration into multi-component research pipelines.

- **Simplicity**: Concise terms (171 words) reduce legal complexity for international academic collaborations.

- **Institutional Acceptance**: Widely pre-approved by university technology transfer offices and research funding agencies (NSF, NIH, DOE).

---

## IV. DOCUMENTATION AND EDUCATIONAL MATERIALS: CREATIVE COMMONS BY 4.0

### 4.1 Applicable Components

- `README.md` — Project overview and quick-start guide
- `/Introductions/` — Comprehensive system documentation
- `/Research Experiments/` — Experimental protocols and research frameworks
- `/paper/` — Manuscript drafts, figures, and supplementary materials
- Tutorial materials, Jupyter notebooks, and example code
- API documentation and technical specifications

### 4.2 License Terms

Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:

**https://creativecommons.org/licenses/by/4.0/**

**You are free to:**
- **Share** — Copy and redistribute the material in any medium or format
- **Adapt** — Remix, transform, and build upon the material for any purpose, even commercially

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

### 4.3 Rationale for CC BY 4.0 Selection

- **Educational Mission**: Promotes scientific education and knowledge dissemination in accordance with open science principles.

- **Attribution Requirement**: Ensures proper academic credit while allowing adaptation for diverse educational contexts.

- **Commercial Compatibility**: Permits commercial educational use (textbooks, online courses) while maintaining attribution.

- **International Standard**: Recognized by major publishers (PLOS, BMC, Nature Communications) for open access scientific content.

---

## V. SCIENTIFIC DATASETS: CREATIVE COMMONS BY-SA 4.0

### 5.1 Applicable Components

- Processed exoplanet parameter datasets
- Derived climate simulation products
- Metabolic pathway annotations
- Training/validation/test data splits
- Benchmark datasets and quality assessment results

### 5.2 License Terms

Licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**:

**https://creativecommons.org/licenses/by-sa/4.0/**

**You are free to:**
- **Share** — Copy and redistribute the material
- **Adapt** — Remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — Must give appropriate credit
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

### 5.3 Rationale for CC BY-SA 4.0 Selection

- **Data Integrity**: ShareAlike provision ensures derivative datasets remain open, preventing proprietary capture of public scientific data.

- **Provenance Tracking**: Attribution requirement maintains data lineage critical for scientific reproducibility.

- **Community Benefit**: Ensures improvements to datasets benefit the broader research community.

- **Standard Practice**: Adopted by major scientific data repositories (Zenodo, Dryad, Figshare) for open research data.

---

## VI. PRIVACY-PRESERVING COMPONENTS: GNU AGPL v3

### 6.1 Applicable Components

- `/customer_data_treatment/` — Privacy-preserving data processing systems:
  - `quantum_enhanced_data_processor.py` — Quantum-resistant encryption
  - `federated_analytics_engine.py` — Federated learning infrastructure
  - Differential privacy mechanisms
  - Homomorphic encryption implementations

### 6.2 License Terms

Licensed under **GNU Affero General Public License v3.0 (AGPL-3.0)**:

**https://www.gnu.org/licenses/agpl-3.0.html**

Key provision: Network copyleft — If you run a modified version on a server and let other users communicate with it there, your server must also allow them to download the source code.

### 6.3 Rationale for AGPL v3 Selection

- **Privacy Protection**: Ensures privacy-preserving technologies remain open and auditable, preventing proprietary capture of user protection mechanisms.

- **Network Copyleft**: Extends copyleft to SaaS deployments, critical for cloud-based AI systems.

- **Transparency**: Mandates source code availability for network services, enabling security audits and privacy verification.

---

## VII. THIRD-PARTY DATA SOURCES AND COMPLIANCE

### 7.1 Scientific Database Licenses

**Public Domain (U.S. Government Works):**
- NASA Exoplanet Archive
- MAST (Mikulski Archive for Space Telescopes)
- NCBI GenBank
- TESS Mission Data

**Creative Commons BY 4.0:**
- UniProt Knowledgebase
- Ensembl Genome Browser

**Creative Commons BY-SA 4.0:**
- GTDB (Genome Taxonomy Database)

**Academic Use Free, Commercial Licensing Required:**
- KEGG (Kyoto Encyclopedia of Genes and Genomes)

**Institutional Access Required:**
- ESO Archive (European Southern Observatory)
- Keck Observatory Archive (KOA)
- Subaru Telescope Archive

### 7.2 Compliance Requirements

Users must:

1. **Verify Licensing**: Check specific dataset licenses before commercial use
2. **Provide Attribution**: Acknowledge all data sources in publications per provider requirements
3. **Respect Restrictions**: Comply with usage restrictions for commercial applications
4. **Maintain Provenance**: Document data lineage and transformations
5. **Export Control**: Comply with U.S. Export Administration Regulations (EAR) for cryptographic components

---

## VIII. CONTRIBUTION LICENSING

### 8.1 Developer Certificate of Origin (DCO)

All contributors must sign off commits using:

```
Signed-off-by: [Full Name] <email@example.com>
```

This certifies compliance with **Developer Certificate of Origin 1.1** (https://developercertificate.org/), affirming that:

1. The contribution is original work or properly licensed
2. Contributor has rights to submit under project license
3. Contribution is provided under project license terms

### 8.2 Contributor License Agreement (CLA)

Contributors grant:

1. **Copyright License**: Right to use, modify, and distribute contributions
2. **Patent License**: License for any patents embodied in contributions
3. **Relicensing Rights**: Right to relicense under compatible open source licenses
4. **Warranty**: Contributions are original or properly licensed

---

## IX. ATTRIBUTION REQUIREMENTS

### 9.1 Academic Publications

When using this platform in academic research, cite as:

```
Astrobiology Platform Research Consortium (2025). Multi-Modal Deep Learning System 
for Exoplanet Habitability Assessment. Version 1.0. 
https://github.com/astrobio-research/astrobio-gen
```

### 9.2 Software Attribution

Include in software distributions:

```
This software incorporates the Astrobiology Research Platform
(https://github.com/astrobio-research/astrobio-gen),
licensed under Apache License 2.0.
Copyright © 2025 Astrobiology Platform Research Consortium.
```

### 9.3 Data Attribution

Acknowledge specific data sources per provider requirements (see Section VII).

---

## X. DISCLAIMER AND LIMITATION OF LIABILITY

**NO WARRANTY**: This software and associated materials are provided "AS IS" without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement.

**LIMITATION OF LIABILITY**: In no event shall the copyright holders or contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

**SCIENTIFIC USE**: Results generated by this platform are for research purposes. Users are responsible for independent validation before making scientific claims or policy recommendations.

---

## XI. CONTACT INFORMATION

**General Licensing Inquiries**: Create GitHub issue with "license" label  
**Commercial Licensing**: Create issue with "commercial" label  
**Academic Partnerships**: Create issue with "academic-partnership" label  
**Security Vulnerabilities**: Report privately via GitHub Security Advisories

**Repository**: https://github.com/astrobio-research/astrobio-gen  
**Documentation**: https://astrobio-gen.readthedocs.io

---

## XII. VERSION HISTORY

**Version 2.0** — October 2, 2025  
- Comprehensive reformatting for academic and scientific rigor
- Enhanced legal clarity and international compliance
- Expanded third-party data source documentation
- Strengthened attribution requirements

**Version 1.0** — July 21, 2025  
- Initial multi-license framework

---

**For the most current license information, always refer to LICENSE.md in the main repository branch.**

**Last Updated**: October 2, 2025  
**Effective Date**: October 2, 2025

