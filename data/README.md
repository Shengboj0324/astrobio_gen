# Data Directory - Secure Storage for Astrobiology Genomics Research

## 🔒 SECURITY NOTICE
This directory contains sensitive scientific data. Follow all security protocols.

## Directory Structure

```
data/
├── raw/                    # Original, immutable source data
│   ├── kegg/              # KEGG pathway database
│   ├── ncbi/              # NCBI genomic data
│   ├── agora2/            # AGORA2 metabolic models
│   ├── 1000g_indices/     # 1000 Genomes indices
│   ├── 1000g_dirlists/    # Directory listings
│   └── stellar_seds/      # Stellar spectral energy distributions
├── interim/               # Intermediate processing stages
│   ├── kegg_edges.csv     # Metabolic reaction networks
│   ├── env_vectors.csv    # Environmental condition vectors
│   └── quality_checks/    # Quality assessment results
├── processed/             # Final, analysis-ready data
│   ├── kegg/              # Processed KEGG pathways
│   ├── ncbi/              # Processed genomic data
│   ├── agora2/            # Processed metabolic models
│   └── quality_reports/   # Data quality reports
├── kegg_graphs/           # Network graph representations (.npz)
├── metadata/              # Data provenance and annotations
├── versions/              # Versioned data snapshots
├── backups/               # Backup storage
└── logs/                  # Processing and access logs
```

## 🛡️ Security Protocols

### 1. Access Control
- **Raw data is READ-ONLY** after initial download
- Use proper file permissions (640 for data files, 750 for directories)
- Never commit raw data to version control
- All data access is logged

### 2. Data Integrity
- All files have MD5/SHA256 checksums
- Integrity checks performed on access
- Automatic corruption detection
- Provenance tracking for all transformations

### 3. Backup Strategy
- Daily incremental backups
- Weekly full backups
- Offsite backup for critical datasets
- Version control for processed data

### 4. Sensitive Data Handling
- Genomic data requires special handling
- PHI/PII screening for human genomic data
- Encrypted storage for sensitive datasets
- Secure deletion procedures

## 📊 Data Types and Sources

### Raw Data Sources
- **KEGG Database**: Metabolic pathway networks
- **NCBI**: Genomic assemblies and annotations
- **AGORA2**: Constraint-based metabolic models
- **1000 Genomes**: Population genomic data
- **Exoplanet Archive**: Planetary system data
- **Stellar Catalogs**: Host star properties

### File Size Guidelines
- Keep individual files < 2GB when possible
- Use compression for large datasets (.gz, .bz2)
- Split large files using standard tools
- Monitor disk usage regularly

## 🔄 Data Processing Workflow

1. **Download**: Raw data → `data/raw/`
2. **Validate**: Quality checks → `data/interim/quality_checks/`
3. **Process**: Cleaned data → `data/interim/`
4. **Finalize**: Analysis-ready → `data/processed/`
5. **Archive**: Long-term storage → `data/versions/`

## 📝 Metadata Requirements

Every dataset must include:
- Source URL and access date
- Processing pipeline version
- Quality assessment results
- Checksum verification
- License and usage terms

## ⚠️ Important Notes

### DO NOT:
- Commit raw data to Git
- Store unencrypted sensitive data
- Modify files in `data/raw/`
- Share data without proper authorization
- Delete data without backup verification

### DO:
- Verify checksums before processing
- Document all processing steps
- Use standard file formats
- Implement proper access controls
- Regular backup verification

## 📞 Data Stewardship Contacts

- **Primary Data Steward**: [Project Lead]
- **Technical Contact**: [Data Engineer]
- **Security Contact**: [Security Officer]

## 🔗 Related Documentation

- [Data Processing Pipeline Guide](../COMPREHENSIVE_DATA_SYSTEM_GUIDE.md)
- [Quality Standards](../DATA_QUALITY_GUIDE.md)
- [Security Policies](../SECURITY_POLICIES.md)

---
**Last Updated**: 2025-01-27
**Version**: 1.0
**Classification**: Internal Use Only 