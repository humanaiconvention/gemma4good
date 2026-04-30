// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title HAICAnchor — on-chain anchoring of HAIC session/training Merkle roots
/// @notice Stores SHA3-256 Merkle roots (computed off-chain by utils/merkle.py)
///         keyed by sessionId. Emits an event for indexers.
/// @dev    Roots are bytes32 — matches the 32-byte SHA3-256 digest exactly.
///         Solidity's `keccak256` is NOT SHA3-256 (NIST). All hashing is done
///         off-chain; this contract only stores and exposes the digests.
contract HAICAnchor {
    enum Kind { Session, Training, Consent }

    struct Anchor {
        bytes32 root;        // SHA3-256 Merkle root
        uint64  blockNumber; // anchor block
        uint64  timestamp;   // block.timestamp at anchor
        address submitter;
        Kind    kind;
        uint32  nodeCount;   // # of leaves in the tree (for audit trail)
    }

    /// sessionId (bytes32) → anchor record. sessionId is caller-defined;
    /// we recommend sha3_256(sessionId_string) so it's also EVM-native bytes32.
    mapping(bytes32 => Anchor) public anchors;

    /// Optional: per-submitter count, useful for rate-limiting at the indexer.
    mapping(address => uint256) public submitterCount;

    event Anchored(
        bytes32 indexed sessionId,
        bytes32 indexed root,
        Kind    indexed kind,
        address submitter,
        uint32  nodeCount,
        uint64  timestamp
    );

    error AlreadyAnchored(bytes32 sessionId);
    error EmptyRoot();
    error EmptySessionId();

    /// @notice Anchor a Merkle root for a session.
    /// @dev    Reverts if sessionId is zero, root is zero, or session already
    ///         anchored. Sessions are append-only — no overwrites.
    function anchor(
        bytes32 sessionId,
        bytes32 root,
        Kind    kind,
        uint32  nodeCount
    ) external {
        if (sessionId == bytes32(0)) revert EmptySessionId();
        if (root == bytes32(0))      revert EmptyRoot();
        if (anchors[sessionId].root != bytes32(0)) {
            revert AlreadyAnchored(sessionId);
        }

        anchors[sessionId] = Anchor({
            root:        root,
            blockNumber: uint64(block.number),
            timestamp:   uint64(block.timestamp),
            submitter:   msg.sender,
            kind:        kind,
            nodeCount:   nodeCount
        });
        unchecked { submitterCount[msg.sender] += 1; }

        emit Anchored(sessionId, root, kind, msg.sender, nodeCount, uint64(block.timestamp));
    }

    /// @notice Read-side helper: returns true iff `expected` matches stored root.
    function verify(bytes32 sessionId, bytes32 expected) external view returns (bool) {
        return anchors[sessionId].root == expected;
    }

    /// @notice Batched view for indexers / dashboards.
    function getAnchor(bytes32 sessionId) external view returns (Anchor memory) {
        return anchors[sessionId];
    }
}
