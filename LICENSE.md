# Astrobiology Platform - Comprehensive Licensing

**Copyright (c) 2025 Astrobiology Platform Contributors**

This project employs multiple licenses to address the diverse nature of its components, data sources, and usage scenarios. Please review the applicable license for each component you're using.

---

## 📋 **License Overview**

| Component | License | Rationale |
|-----------|---------|-----------|
| **Core Research Code** | Apache 2.0 | Open collaboration with patent protection |
| **Training Infrastructure** | Apache 2.0 | Enterprise-friendly with attribution |
| **AI Models & Architectures** | Apache 2.0 | Research and commercial use |
| **Scientific Data Processing** | MIT | Maximum compatibility with research institutions |
| **Customer Data Treatment** | AGPL v3 | Ensures privacy-preserving derivatives remain open |
| **LLM Integration** | Apache 2.0 + Model-specific | Respects underlying model licenses |
| **Documentation & Examples** | CC BY 4.0 | Promotes scientific education and sharing |
| **Scientific Datasets** | CC BY-SA 4.0 | Attribution with share-alike for data integrity |
| **API & Web Services** | Apache 2.0 | Enterprise integration friendly |
| **Configuration Files** | CC0 1.0 | Public domain for maximum reuse |

---

## 🔬 **Core Research Code - Apache License 2.0**

**Applies to:**
- `/models/` - All neural network architectures
- `/training/` - Training orchestrator and modules
- `/utils/` - System utilities and diagnostics
- `/validation/` - System validation frameworks
- `/monitoring/` - Real-time monitoring systems
- `train.py`, `train_enhanced_cube.py` - Main training scripts

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**Why Apache 2.0:**
- Provides explicit patent grant protection
- Enterprise-friendly for commercial use
- Requires attribution and license notice
- Compatible with most other open source licenses
- Preferred by many research institutions and companies

---

## 📊 **Scientific Data Processing - MIT License**

**Applies to:**
- `/data_build/` - Data management and processing systems
- `/datamodules/` - PyTorch Lightning data modules
- `/scripts/` - Data acquisition and processing scripts
- `run_comprehensive_data_system.py`

```
MIT License

Copyright (c) 2025 Astrobiology Platform Contributors

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

**Why MIT:**
- Maximum compatibility with research institutions
- Minimal restrictions for scientific collaboration
- Widely understood and accepted in academia
- Compatible with proprietary research tools

---

## 🔐 **Customer Data Treatment - AGPL v3**

**Applies to:**
- `/customer_data_treatment/` - Quantum-enhanced data processing
- Privacy-preserving federated analytics
- Homomorphic encryption implementations
- Differential privacy mechanisms

```
GNU AFFERO GENERAL PUBLIC LICENSE
Version 3, 19 November 2007

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

**Why AGPL v3:**
- Ensures privacy-preserving technologies remain open
- Prevents proprietary capture of customer data protection mechanisms
- Network copyleft applies to SaaS deployments
- Protects users' privacy rights through code transparency

---

## 🤖 **LLM Integration - Apache 2.0 + Model-Specific**

**Applies to:**
- `/models/peft_llm_integration.py`
- LLM training and inference modules
- Knowledge retrieval systems
- Voice synthesis integration

**Base License:** Apache 2.0 (see above)

**Additional Model License Compliance:**

### Microsoft DialoGPT
- **License:** MIT License
- **Attribution Required:** Yes
- **Commercial Use:** Permitted
- **Model Citation:** Microsoft Conversational AI Research

### Hugging Face Transformers
- **License:** Apache 2.0
- **Patent Grant:** Included
- **Attribution Required:** Yes

**Note:** Users must comply with the specific licenses of any LLMs they use with this system.

---

## 📚 **Documentation & Examples - Creative Commons BY 4.0**

**Applies to:**
- `README.md`
- `/Introductions/` - All documentation files
- `/examples/` - Example code and notebooks
- Tutorial materials and guides
- Scientific methodology documentation

```
Creative Commons Attribution 4.0 International License

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material
for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license,
  and indicate if changes were made. You may do so in any reasonable manner,
  but not in any way that suggests the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological
measures that legally restrict others from doing anything the license permits.
```

**Why CC BY 4.0:**
- Promotes scientific education and knowledge sharing
- Allows adaptation for different research contexts
- Requires attribution to maintain scientific credit
- Compatible with commercial educational use

---

## 📈 **Scientific Datasets - Creative Commons BY-SA 4.0**

**Applies to:**
- Processed scientific datasets
- Derived data products
- Training data annotations
- Benchmark datasets
- Quality assessment results

```
Creative Commons Attribution-ShareAlike 4.0 International License

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material
for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit
- ShareAlike — If you remix, transform, or build upon the material,
  you must distribute your contributions under the same license as the original.
```

**Why CC BY-SA 4.0:**
- Ensures scientific data improvements benefit the community
- Maintains data provenance and attribution
- Prevents proprietary capture of public scientific data
- Standard for open scientific datasets

---

## ⚙️ **Configuration Files - CC0 1.0 (Public Domain)**

**Applies to:**
- `/config/` - All configuration files
- `requirements.txt`, `requirements_llm.txt`
- Setup and installation scripts
- Environment configuration files

```
CC0 1.0 Universal (CC0 1.0) Public Domain Dedication

The person who associated a work with this deed has dedicated the work to the
public domain by waiving all of his or her rights to the work worldwide under
copyright law, including all related and neighboring rights, to the extent
allowed by law.

You can copy, modify, distribute and perform the work, even for commercial
purposes, all without asking permission.
```

**Why CC0:**
- Maximum reuse for configuration and setup
- No barriers to adaptation for different environments
- Standard practice for configuration files
- Facilitates easy integration and deployment

---

## 🌐 **Third-Party Data Source Licenses**

### Scientific Databases
- **KEGG Database:** Academic use free, commercial licensing required
- **NCBI Data:** Public domain (U.S. Government work)
- **NASA Exoplanet Archive:** Public domain (U.S. Government work)
- **UniProt:** CC BY 4.0
- **JGI Genomes:** Varies by dataset, generally open for research
- **GTDB:** CC BY-SA 4.0

### Compliance Requirements
Users must:
1. **Check specific dataset licenses** before commercial use
2. **Provide appropriate attribution** for all data sources
3. **Respect usage restrictions** for commercial applications
4. **Acknowledge data providers** in publications

---

## 🏢 **Commercial Licensing Options**

### Dual Licensing Available
For organizations requiring different licensing terms:

**Enterprise License:**
- Proprietary licensing available for components under Apache 2.0
- Custom license terms for specific use cases
- Priority support and warranty options
- Contact: [Commercial Licensing Contact]

**Research Institution License:**
- Special terms for academic and research institutions
- Bulk licensing for educational use
- Collaboration and partnership opportunities
- Contact: [Academic Partnerships Contact]

---

## 🔒 **Export Control and Security**

### U.S. Export Administration Regulations (EAR)
- Cryptographic components may be subject to export controls
- Users responsible for compliance with local export laws
- Open source exception generally applies

### Security Considerations
- Customer data treatment components use advanced cryptography
- Users must comply with local privacy and security regulations
- GDPR, CCPA, and other privacy law compliance required

---

## 📜 **License Compatibility Matrix**

| License Combination | Compatible | Notes |
|---------------------|------------|-------|
| Apache 2.0 + MIT | ✅ Yes | Standard combination |
| Apache 2.0 + AGPL v3 | ⚠️ Limited | AGPL applies to combined work |
| MIT + CC BY 4.0 | ✅ Yes | Common for code + documentation |
| All Open Source | ✅ Yes | With proper attribution |
| Commercial + Open | ✅ Yes | Via dual licensing |

---

## 🤝 **Contributing License Agreement**

### Contributor License Agreement (CLA)
Contributors grant:
1. **Copyright license** to use, modify, and distribute contributions
2. **Patent license** for any patents in contributions
3. **Right to relicense** under compatible open source licenses
4. **Warranty** that contributions are original or properly licensed

### Developer Certificate of Origin (DCO)
All commits must include:
```
Signed-off-by: [Your Name] <your.email@example.com>
```

This certifies compliance with [Developer Certificate of Origin 1.1](https://developercertificate.org/).

---

## 📞 **Contact Information**

### Licensing Questions
- **General Licensing:** [License Contact]
- **Commercial Licensing:** [Commercial Contact]
- **Academic Partnerships:** [Academic Contact]
- **Legal Compliance:** [Legal Contact]

### Repository Issues
- **License Compliance Issues:** File issue with "license" label
- **Attribution Questions:** Use "attribution" label
- **Commercial Inquiries:** Use "commercial" label

---

## 📅 **License Updates**

**Version:** 2.0  
**Effective Date:** July 21, 2025  
**Last Updated:** July 21, 2025

### Change Log
- **v2.0 (July 2025):** Comprehensive multi-license framework
- **v1.0 (Initial):** MIT License only

### Future Updates
License updates will be announced via:
- Repository releases
- Project website
- Contributor mailing list
- Major version tags

---

**For the most current license information, always refer to the LICENSE.md file in the main repository branch.**