// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {HAICAnchor} from "../contracts/HAICAnchor.sol";

/// @notice Deploy HAICAnchor.
/// Usage:
///   forge script script/Deploy.s.sol --rpc-url sepolia --broadcast --verify
contract Deploy is Script {
    function run() external returns (HAICAnchor a) {
        uint256 pk = vm.envUint("DEPLOYER_PRIVATE_KEY");
        vm.startBroadcast(pk);
        a = new HAICAnchor();
        vm.stopBroadcast();
        console.log("HAICAnchor deployed at:", address(a));
    }
}
