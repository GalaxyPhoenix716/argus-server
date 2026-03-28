import { defineConfig } from "hardhat/config";

export default defineConfig({
  solidity: {
    version: "0.8.28",
  },
  paths: {
    sources:   "./contracts",
    artifacts: "./artifacts",
  },
  networks: {
    localhost: {
      type: "http",
      url:  "http://127.0.0.1:8545",
    },
  },
});
