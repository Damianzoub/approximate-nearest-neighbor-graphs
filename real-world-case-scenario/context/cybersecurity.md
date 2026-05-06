# Cybersecurity

Cybersecurity encompasses the practices, technologies, and processes designed to protect computer systems, networks, software, and data from attack, damage, or unauthorised access. As society becomes increasingly digital, cybersecurity has evolved from a technical concern to a fundamental pillar of national security and economic stability.

## Threat Landscape

Cyberattacks take many forms. Malware — malicious software including viruses, worms, trojans, ransomware, and spyware — is delivered through phishing emails, malicious downloads, or compromised websites. Ransomware encrypts victims' data and demands payment for decryption keys; attacks on hospitals, pipelines, and governments have caused billions in damages and even loss of life.

Advanced Persistent Threats (APTs) are sophisticated, often state-sponsored campaigns that infiltrate target networks and remain undetected for months or years. Social engineering attacks exploit human psychology rather than technical vulnerabilities — phishing, pretexting, and spear-phishing remain among the most effective attack vectors.

Supply chain attacks compromise software or hardware components used by many downstream targets, as demonstrated by the SolarWinds and Log4Shell incidents. Zero-day vulnerabilities — previously unknown flaws — are highly valued on both legitimate bug bounty markets and criminal forums.

## Cryptography and Secure Communications

Cryptography is the mathematical foundation of secure communications. Symmetric encryption (e.g., AES) uses the same key for encryption and decryption, making key distribution a challenge. Asymmetric encryption (e.g., RSA, ECC) uses public-private key pairs, enabling secure key exchange and digital signatures over untrusted networks.

The TLS (Transport Layer Security) protocol secures most internet traffic, combining asymmetric cryptography for key exchange with symmetric encryption for bulk data transfer. Digital certificates, issued by Certificate Authorities (CAs), bind public keys to identities, enabling authentication of websites and services.

Hash functions (SHA-256, SHA-3) produce fixed-length digests of arbitrary data, enabling data integrity verification, password storage, and blockchain construction. Modern password storage uses adaptive hashing algorithms like bcrypt, scrypt, or Argon2 that are deliberately slow to resist brute-force attacks.

## Defence Strategies

A defence-in-depth strategy layers multiple security controls — firewalls, intrusion detection systems, endpoint protection, network segmentation, and multi-factor authentication — so that no single failure compromises the entire system. The Zero Trust model assumes that no user or device inside or outside the network is inherently trusted; every access request must be verified.

Security Information and Event Management (SIEM) systems aggregate and analyse logs from across an organisation's infrastructure, enabling detection of anomalies and coordinated attacks. Vulnerability management involves continuously scanning systems, prioritising patches, and remediating weaknesses before attackers exploit them.

## Post-Quantum Cryptography

The advent of capable quantum computers threatens current public-key cryptographic algorithms: Shor's algorithm can break RSA and ECC. NIST has standardised post-quantum cryptographic algorithms — including CRYSTALS-Kyber for key encapsulation and CRYSTALS-Dilithium for digital signatures — based on lattice problems believed to be hard for both classical and quantum computers.

Organisations must begin "crypto-agility" migrations now, replacing vulnerable algorithms in long-lived systems before cryptographically relevant quantum computers arrive, estimated conservatively within 10-15 years.

## AI in Cybersecurity

AI is used both offensively and defensively in cybersecurity. Defenders use machine learning to detect malware, identify phishing emails, and detect anomalous user behaviour indicative of account compromise. Attackers use AI to generate more convincing phishing content, automate vulnerability discovery, and create polymorphic malware that evades signature-based detection.
