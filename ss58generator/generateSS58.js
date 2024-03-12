const { cryptoWaitReady, encodeAddress, randomAsU8a } = require('@polkadot/util-crypto');

const generateSS58Address = async () => {
    await cryptoWaitReady();
    const seed = randomAsU8a(32);
    const address = encodeAddress(seed);
    console.log(address);
    console.log(seed)
};

generateSS58Address();

