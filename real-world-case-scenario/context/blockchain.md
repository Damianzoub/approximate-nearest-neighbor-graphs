# Blockchain and Distributed Ledger Technology

A blockchain is a distributed ledger — a database shared and synchronised across many computers — in which data is stored in linked blocks secured by cryptographic hashes. Once a block is added to the chain, altering it would require recalculating the hashes of all subsequent blocks, making tampering computationally prohibitive. Blockchains enable trustless transactions between parties who do not know or trust each other.

## How Blockchains Work

Each block contains a list of transactions, a timestamp, and a cryptographic hash of the previous block. The hash of each block depends on its contents and the previous block's hash, creating an immutable chain. Any attempt to alter a historical block would produce a different hash, breaking the chain and alerting honest participants.

Consensus mechanisms determine which participant gets to add the next block and ensure all participants agree on the state of the ledger. Proof of Work (PoW), used by Bitcoin, requires miners to solve computationally expensive puzzles — consuming significant energy. Proof of Stake (PoS), used by Ethereum after "The Merge" in 2022, selects validators based on the cryptocurrency they lock up as collateral, dramatically reducing energy consumption.

## Bitcoin and Cryptocurrency

Bitcoin, introduced by the pseudonymous Satoshi Nakamoto in 2008, was the first successful cryptocurrency — a decentralised digital currency without a central bank or administrator. Bitcoin transactions are verified by a decentralised network of nodes and recorded on a public blockchain. The total supply is capped at 21 million bitcoins, enforced by the protocol itself.

Cryptocurrencies offer permissionless, censorship-resistant, borderless value transfer but have faced criticism for price volatility, use in illicit transactions, and energy consumption (for PoW chains). Stablecoins — cryptocurrencies pegged to a stable asset like the US dollar — attempt to provide the benefits of cryptocurrency without price volatility.

## Smart Contracts and Ethereum

Ethereum, launched in 2015, extended blockchain beyond currency by introducing smart contracts — self-executing programmes stored on the blockchain that automatically enforce agreements when conditions are met. Smart contracts enable decentralised applications (dApps) that run exactly as programmed, without downtime, censorship, or third-party interference.

Smart contracts are written in languages like Solidity. They power decentralised finance (DeFi) applications — lending, borrowing, trading, and yield farming — entirely on-chain without traditional financial intermediaries. Non-Fungible Tokens (NFTs), unique digital assets recorded on a blockchain, sparked a wave of digital art and collectibles but also significant speculation.

## Decentralised Finance (DeFi)

DeFi protocols replicate financial services — exchanges, lending markets, derivatives — using smart contracts on public blockchains. Automated Market Makers (AMMs), such as Uniswap, allow users to trade tokens directly from liquidity pools rather than with a counterparty. Lending protocols like Aave and Compound allow users to borrow and lend cryptocurrencies with interest rates determined algorithmically.

DeFi grew to over $100 billion in total value locked at its peak, but also suffered significant exploits and hacks due to smart contract vulnerabilities. Regulatory uncertainty remains a major challenge.

## Enterprise Blockchain and Supply Chain

Beyond public cryptocurrencies, private and permissioned blockchains are used in enterprise settings. Hyperledger Fabric, developed by the Linux Foundation, enables businesses to deploy private blockchains where participants are known and trusted. Applications include supply chain tracking — verifying product provenance from farm to store — trade finance, cross-border payments, and digital identity.

The promise of enterprise blockchain is selective sharing of verified data between organisations without a central data custodian, reducing reconciliation costs and enabling new forms of inter-organisational collaboration.
