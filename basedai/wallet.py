""" Implementation of the wallet class, which manages balances with staking and transfer. Also manages computekey and personalkey.
"""

# The MIT License (MIT)
# Copyright © 2024 Saul Finney

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import copy
import argparse
import basedai
import hashlib
from termcolor import colored
from substrateinterface import Keypair
from typing import Optional, Union, List, Tuple, Dict, overload
from basedai.utils import is_valid_basedai_address_or_public_key
from substrateinterface.utils.ss58 import ss58_decode

def ss58_to_ethereum(ss58_address):
    public_key = ss58_decode(ss58_address)

    keccak_hash = hashlib.sha3_256(bytes.fromhex(public_key)).digest()
    eth_address = keccak_hash[-20:]

    return '0x' + eth_address.hex()

def display_mnemonic_msg(keypair: Keypair, key_type: str):
    """
    Display the mnemonic and a warning message to keep the mnemonic safe.

    Args:
        keypair (Keypair): Keypair object.
        key_type (str): Type of the key (personalkey or computekey).
    """
    mnemonic = keypair.mnemonic
    mnemonic_green = colored(mnemonic, "green")
    print(
        colored(
            "\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone "
            "who has possession of this mnemonic can use it to regenerate the key and access your tokens. \n",
            "red",
        )
    )
    print("The mnemonic to the new {} is:\n\n{}\n".format(key_type, mnemonic_green))
    print(
        "You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:"
    )
    print("basedcli w regen_{} --mnemonic {}".format(key_type, mnemonic))
    print("")


class wallet:
    """
    The wallet class in the Basedai framework handles wallet functionality, crucial for participating in the Basedai network.

    It manages two types of keys: personalkey and computekey, each serving different purposes in network operations. Each wallet contains a personalkey and a computekey.

    The personalkey is the user's primary key for holding stake in their wallet and is the only way that users
    can access Based. Personalkeys can hold tokens and should be encrypted on your device.

    The personalkey is the primary key used for securing the wallet's stake in the BasedAI Network (Based) and
    is critical for financial transactions like staking and unstaking tokens. It's recommended to keep the
    personalkey encrypted and secure, as it holds the actual tokens.

    The computekey, in contrast, is used for operational tasks like subscribing to and setting weights in the
    network. It's linked to the personalkey through the stem and does not directly hold tokens, thereby
    offering a safer way to interact with the network during regular operations.

    Args:
        name (str): The name of the wallet, used to identify it among possibly multiple wallets.
        path (str): File system path where wallet keys are stored.
        computekey_str (str): String identifier for the computekey.
        _computekey, _personalkey, _personalkeypub (basedai.Keypair): Internal representations of the computekey and personalkey.

    Methods:
        create_if_non_existent, create, recreate: Methods to handle the creation of wallet keys.
        get_personalkey, get_computekey, get_personalkeypub: Methods to retrieve specific keys.
        set_personalkey, set_computekey, set_personalkeypub: Methods to set or update keys.
        computekey_file, personalkey_file, personalkeypub_file: Properties that return respective key file objects.
        regenerate_personalkey, regenerate_computekey, regenerate_personalkeypub: Methods to regenerate keys from different sources.
        config, help, add_args: Utility methods for configuration and assistance.

    The wallet class is a fundamental component for users to interact securely with the Basedai network, facilitating both operational tasks and transactions involving value transfer across the network.

    Example Usage::

        # Create a new wallet with default personalkey and computekey names
        my_wallet = wallet()

        # Access computekey and personalkey
        computekey = my_wallet.get_computekey()
        personalkey = my_wallet.get_personalkey()

        # Set a new personalkey
        my_wallet.new_personalkey(n_words=24) # number of seed words to use

        # Update wallet computekey
        my_wallet.set_computekey(new_computekey)

        # Print wallet details
        print(my_wallet)

        # Access personalkey property, must use password to unlock
        my_wallet.personalkey
    """

    @classmethod
    def config(cls) -> "basedai.config":
        """
        Get config from the argument parser.

        Returns:
            basedai.config: Config object.
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        return basedai.config(parser, args=[])

    @classmethod
    def help(cls):
        """
        Print help to stdout.
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """
        Accept specific arguments from parser.

        Args:
            parser (argparse.ArgumentParser): Argument parser object.
            prefix (str): Argument prefix.
        """
        prefix_str = "" if prefix == None else prefix + "."
        try:
            default_name = os.getenv("BT_WALLET_NAME") or "default"
            default_computekey = os.getenv("BT_WALLET_NAME") or "default"
            default_path = os.getenv("BT_WALLET_PATH") or "~/.basedai/wallets/"
            parser.add_argument(
                "--" + prefix_str + "wallet.name",
                required=False,
                default=default_name,
                help="The name of the wallet to unlock for running basedai "
                "(name mock is reserved for mocking this wallet)",
            )
            parser.add_argument(
                "--" + prefix_str + "wallet.computekey",
                required=False,
                default=default_computekey,
                help="The name of the wallet's computekey.",
            )
            parser.add_argument(
                "--" + prefix_str + "wallet.path",
                required=False,
                default=default_path,
                help="The path to your basedai wallets",
            )
        except argparse.ArgumentError as e:
            pass

    def __init__(
        self,
        name: str = None,
        computekey: str = None,
        path: str = None,
        config: "basedai.config" = None,
    ):
        r"""
        Initialize the basedai wallet object containing a hot and personalkey.

        Args:
            name (str, optional): The name of the wallet to unlock for running basedai. Defaults to ``default``.
            computekey (str, optional): The name of computekey used to running the miner. Defaults to ``default``.
            path (str, optional): The path to your basedai wallets. Defaults to ``~/.basedai/wallets/``.
            config (basedai.config, optional): basedai.wallet.config(). Defaults to ``None``.
        """
        # Fill config from passed args using command line defaults.
        if config is None:
            config = wallet.config()
        self.config = copy.deepcopy(config)
        self.config.wallet.name = name or self.config.wallet.get(
            "name", basedai.defaults.wallet.name
        )
        self.config.wallet.computekey = computekey or self.config.wallet.get(
            "computekey", basedai.defaults.wallet.computekey
        )
        self.config.wallet.path = path or self.config.wallet.get(
            "path", basedai.defaults.wallet.path
        )

        self.name = self.config.wallet.name
        self.path = self.config.wallet.path
        self.computekey_str = self.config.wallet.computekey

        self._computekey = None
        self._personalkey = None
        self._personalkeypub = None

    def __str__(self):
        """
        Returns the string representation of the Wallet object.

        Returns:
            str: The string representation.
        """
        return "wallet({}, {}, {})".format(self.name, self.computekey_str, self.path)

    def __repr__(self):
        """
        Returns the string representation of the wallet object.

        Returns:
            str: The string representation.
        """
        return self.__str__()

    def create_if_non_existent(
        self, personalkey_use_password: bool = True, computekey_use_password: bool = False
    ) -> "wallet":
        """
        Checks for existing personalkeypub and computekeys, and creates them if non-existent.

        Args:
            personalkey_use_password (bool, optional): Whether to use a password for personalkey. Defaults to ``True``.
            computekey_use_password (bool, optional): Whether to use a password for computekey. Defaults to ``False``.

        Returns:
            wallet: The wallet object.
        """
        return self.create(personalkey_use_password, computekey_use_password)

    def create(
        self, personalkey_use_password: bool = True, computekey_use_password: bool = False
    ) -> "wallet":
        """
        Checks for existing personalkeypub and computekeys, and creates them if non-existent.

        Args:
            personalkey_use_password (bool, optional): Whether to use a password for personalkey. Defaults to ``True``.
            computekey_use_password (bool, optional): Whether to use a password for computekey. Defaults to ``False``.

        Returns:
            wallet: The wallet object.
        """
        # ---- Setup Wallet. ----
        if (
            not self.personalkey_file.exists_on_device()
            and not self.personalkeypub_file.exists_on_device()
        ):
            self.create_new_personalkey(n_words=12, use_password=personalkey_use_password)
        if not self.computekey_file.exists_on_device():
            self.create_new_computekey(n_words=12, use_password=computekey_use_password)
        return self

    def recreate(
        self, personalkey_use_password: bool = True, computekey_use_password: bool = False
    ) -> "wallet":
        """
        Checks for existing personalkeypub and computekeys and creates them if non-existent.

        Args:
            personalkey_use_password (bool, optional): Whether to use a password for personalkey. Defaults to ``True``.
            computekey_use_password (bool, optional): Whether to use a password for computekey. Defaults to ``False``.

        Returns:
            wallet: The wallet object.
        """
        # ---- Setup Wallet. ----
        self.create_new_personalkey(n_words=12, use_password=personalkey_use_password)
        self.create_new_computekey(n_words=12, use_password=computekey_use_password)
        return self

    @property
    def computekey_file(self) -> "basedai.keyfile":
        """
        Property that returns the computekey file.

        Returns:
            basedai.keyfile: The computekey file.
        """
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        computekey_path = os.path.join(wallet_path, "computekeys", self.computekey_str)
        return basedai.keyfile(path=computekey_path)

    @property
    def personalkey_file(self) -> "basedai.keyfile":
        """
        Property that returns the personalkey file.

        Returns:
            basedai.keyfile: The personalkey file.
        """
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        personalkey_path = os.path.join(wallet_path, "personalkey")
        return basedai.keyfile(path=personalkey_path)

    @property
    def personalkeypub_file(self) -> "basedai.keyfile":
        """
        Property that returns the personalkeypub file.

        Returns:
            basedai.keyfile: The personalkeypub file.
        """
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        personalkeypub_path = os.path.join(wallet_path, "personalkeypub.txt")
        return basedai.keyfile(path=personalkeypub_path)

    def set_computekey(
        self,
        keypair: "basedai.Keypair",
        encrypt: bool = False,
        overwrite: bool = False,
    ) -> "basedai.keyfile":
        """
        Sets the computekey for the wallet.

        Args:
            keypair (basedai.Keypair): The computekey keypair.
            encrypt (bool, optional): Whether to encrypt the computekey. Defaults to ``False``.
            overwrite (bool, optional): Whether to overwrite an existing computekey. Defaults to ``False``.

        Returns:
            basedai.keyfile: The computekey file.
        """
        self._computekey = keypair
        self.computekey_file.set_keypair(keypair, encrypt=encrypt, overwrite=overwrite)

    def set_personalkeypub(
        self,
        keypair: "basedai.Keypair",
        encrypt: bool = False,
        overwrite: bool = False,
    ) -> "basedai.keyfile":
        """
        Sets the personalkeypub for the wallet.

        Args:
            keypair (basedai.Keypair): The personalkeypub keypair.
            encrypt (bool, optional): Whether to encrypt the personalkeypub. Defaults to ``False``.
            overwrite (bool, optional): Whether to overwrite an existing personalkeypub. Defaults to ``False``.

        Returns:
            basedai.keyfile: The personalkeypub file.
        """
        self._personalkeypub = basedai.Keypair(ss58_address=keypair.ss58_address)
        self.personalkeypub_file.set_keypair(
            self._personalkeypub, encrypt=encrypt, overwrite=overwrite
        )

    def set_personalkey(
        self,
        keypair: "basedai.Keypair",
        encrypt: bool = True,
        overwrite: bool = False,
    ) -> "basedai.keyfile":
        """
        Sets the personalkey for the wallet.

        Args:
            keypair (basedai.Keypair): The personalkey keypair.
            encrypt (bool, optional): Whether to encrypt the personalkey. Defaults to ``True``.
            overwrite (bool, optional): Whether to overwrite an existing personalkey. Defaults to ``False``.

        Returns:
            basedai.keyfile: The personalkey file.
        """
        self._personalkey = keypair
        self.personalkey_file.set_keypair(
            self._personalkey, encrypt=encrypt, overwrite=overwrite
        )

    def get_personalkey(self, password: str = None) -> "basedai.Keypair":
        """
        Gets the personalkey from the wallet.

        Args:
            password (str, optional): The password to decrypt the personalkey. Defaults to ``None``.

        Returns:
            basedai.Keypair: The personalkey keypair.
        """
        return self.personalkey_file.get_keypair(password=password)

    def get_computekey(self, password: str = None) -> "basedai.Keypair":
        """
        Gets the computekey from the wallet.

        Args:
            password (str, optional): The password to decrypt the computekey. Defaults to ``None``.

        Returns:
            basedai.Keypair: The computekey keypair.
        """
        return self.computekey_file.get_keypair(password=password)

    def get_personalkeypub(self, password: str = None) -> "basedai.Keypair":
        """
        Gets the personalkeypub from the wallet.

        Args:
            password (str, optional): The password to decrypt the personalkeypub. Defaults to ``None``.

        Returns:
            basedai.Keypair: The personalkeypub keypair.
        """
        return self.personalkeypub_file.get_keypair(password=password)

    @property
    def computekey(self) -> "basedai.Keypair":
        r"""Loads the computekey from wallet.path/wallet.name/computekeys/wallet.computekey or raises an error.

        Returns:
            computekey (Keypair):
                computekey loaded from config arguments.
        Raises:
            KeyFileError: Raised if the file is corrupt of non-existent.
            CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._computekey == None:
            self._computekey = self.computekey_file.keypair
        return self._computekey

    @property
    def personalkey(self) -> "basedai.Keypair":
        r"""Loads the computekey from wallet.path/wallet.name/personalkey or raises an error.

        Returns:
            personalkey (Keypair): personalkey loaded from config arguments.
        Raises:
            KeyFileError: Raised if the file is corrupt of non-existent.
            CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._personalkey == None:
            self._personalkey = self.personalkey_file.keypair
        return self._personalkey

    @property
    def personalkeypub(self) -> "basedai.Keypair":
        r"""Loads the personalkeypub from wallet.path/wallet.name/personalkeypub.txt or raises an error.

        Returns:
            personalkeypub (Keypair): personalkeypub loaded from config arguments.
        Raises:
            KeyFileError: Raised if the file is corrupt of non-existent.
            CryptoKeyError: Raised if the user enters an incorrect password for an encrypted keyfile.
        """
        if self._personalkeypub == None:
            self._personalkeypub = self.personalkeypub_file.keypair
        return self._personalkeypub

    def create_personalkey_from_uri(
        self,
        uri: str,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Creates personalkey from suri string, optionally encrypts it with the user-provided password.

        Args:
            uri: (str, required):
                URI string to use i.e., ``/Alice`` or ``/Bob``.
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determines if this operation overwrites the personalkey under the same path ``<wallet path>/<wallet name>/personalkey``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created personalkey.
        """
        keypair = Keypair.create_from_uri(uri)
        if not suppress:
            display_mnemonic_msg(keypair, "personalkey")
        self.set_personalkey(keypair, encrypt=use_password, overwrite=overwrite)
        self.set_personalkeypub(keypair, overwrite=overwrite)
        return self

    def create_computekey_from_uri(
        self,
        uri: str,
        use_password: bool = False,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Creates computekey from suri string, optionally encrypts it with the user-provided password.

        Args:
            uri: (str, required):
                URI string to use i.e., ``/Alice`` or ``/Bob``
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determines if this operation overwrites the computekey under the same path ``<wallet path>/<wallet name>/computekeys/<computekey>``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created computekey.
        """
        keypair = Keypair.create_from_uri(uri)
        if not suppress:
            display_mnemonic_msg(keypair, "computekey")
        self.set_computekey(keypair, encrypt=use_password, overwrite=overwrite)
        return self

    def new_personalkey(
        self,
        n_words: int = 12,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Creates a new personalkey, optionally encrypts it with the user-provided password and saves to disk.

        Args:
            n_words: (int, optional):
                Number of mnemonic words to use.
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determines if this operation overwrites the personalkey under the same path ``<wallet path>/<wallet name>/personalkey``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created personalkey.
        """
        self.create_new_personalkey(n_words, use_password, overwrite, suppress)

    def create_new_personalkey(
        self,
        n_words: int = 12,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Creates a new personalkey, optionally encrypts it with the user-provided password and saves to disk.

        Args:
            n_words: (int, optional):
                Number of mnemonic words to use.
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determines if this operation overwrites the personalkey under the same path ``<wallet path>/<wallet name>/personalkey``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created personalkey.
        """
        mnemonic = Keypair.generate_mnemonic(n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        if not suppress:
            display_mnemonic_msg(keypair, "personalkey")
        self.set_personalkey(keypair, encrypt=use_password, overwrite=overwrite)
        self.set_personalkeypub(keypair, overwrite=overwrite)
        return self

    def new_computekey(
        self,
        n_words: int = 12,
        use_password: bool = False,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Creates a new computekey, optionally encrypts it with the user-provided password and saves to disk.

        Args:
            n_words: (int, optional):
                Number of mnemonic words to use.
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determines if this operation overwrites the computekey under the same path ``<wallet path>/<wallet name>/computekeys/<computekey>``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created computekey.
        """
        self.create_new_computekey(n_words, use_password, overwrite, suppress)

    def create_new_computekey(
        self,
        n_words: int = 12,
        use_password: bool = False,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Creates a new computekey, optionally encrypts it with the user-provided password and saves to disk.

        Args:
            n_words: (int, optional):
                Number of mnemonic words to use.
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Will this operation overwrite the computekey under the same path <wallet path>/<wallet name>/computekeys/<computekey>
        Returns:
            wallet (basedai.wallet):
                This object with newly created computekey.
        """
        mnemonic = Keypair.generate_mnemonic(n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        if not suppress:
            display_mnemonic_msg(keypair, "computekey")
        self.set_computekey(keypair, encrypt=use_password, overwrite=overwrite)
        return self

    def regenerate_personalkeypub(
        self,
        ss58_address: Optional[str] = None,
        public_key: Optional[Union[str, bytes]] = None,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        """Regenerates the personalkeypub from the passed ``ss58_address`` or public_key and saves the file. Requires either ``ss58_address`` or public_key to be passed.

        Args:
            ss58_address: (str, optional):
                Address as ``ss58`` string.
            public_key: (str | bytes, optional):
                Public key as hex string or bytes.
            overwrite (bool, optional) (default: False):
                Determins if this operation overwrites the personalkeypub (if exists) under the same path ``<wallet path>/<wallet name>/personalkeypub``.
        Returns:
            wallet (basedai.wallet):
                Newly re-generated wallet with personalkeypub.

        """
        if ss58_address is None and public_key is None:
            raise ValueError("Either ss58_address or public_key must be passed")

        if not is_valid_basedai_address_or_public_key(
            ss58_address if ss58_address is not None else public_key
        ):
            raise ValueError(
                f"Invalid {'ss58_address' if ss58_address is not None else 'public_key'}"
            )

        if ss58_address is not None:
            ss58_format = basedai.utils.get_ss58_format(ss58_address)
            keypair = Keypair(
                ss58_address=ss58_address,
                public_key=public_key,
                ss58_format=ss58_format,
            )
        else:
            keypair = Keypair(
                ss58_address=ss58_address,
                public_key=public_key,
                ss58_format=basedai.__ss58_format__,
            )

        # No need to encrypt the public key
        self.set_personalkeypub(keypair, overwrite=overwrite)

        return self

    # Short name for regenerate_personalkeypub
    regen_personalkeypub = regenerate_personalkeypub

    @overload
    def regenerate_personalkey(
        self,
        mnemonic: Optional[Union[list, str]] = None,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        ...

    @overload
    def regenerate_personalkey(
        self,
        seed: Optional[str] = None,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        ...

    @overload
    def regenerate_personalkey(
        self,
        json: Optional[Tuple[Union[str, Dict], str]] = None,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        ...

    def regenerate_personalkey(
        self,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
        **kwargs,
    ) -> "wallet":
        """Regenerates the personalkey from the passed mnemonic or seed, or JSON encrypts it with the user's password and saves the file.

        Args:
            mnemonic: (Union[list, str], optional):
                Key mnemonic as list of words or string space separated words.
            seed: (str, optional):
                Seed as hex string.
            json: (Tuple[Union[str, Dict], str], optional):
                Restore from encrypted JSON backup as ``(json_data: Union[str, Dict], passphrase: str)``
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determines if this operation overwrites the personalkey under the same path ``<wallet path>/<wallet name>/personalkey``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created personalkey.

        Note:
            Uses priority order: ``mnemonic > seed > json``.

        """
        if len(kwargs) == 0:
            raise ValueError("Must pass either mnemonic, seed, or json")

        # Get from kwargs
        mnemonic = kwargs.get("mnemonic", None)
        seed = kwargs.get("seed", None)
        json = kwargs.get("json", None)

        if mnemonic is None and seed is None and json is None:
            raise ValueError("Must pass either mnemonic, seed, or json")
        if mnemonic is not None:
            if isinstance(mnemonic, str):
                mnemonic = mnemonic.split()
            if len(mnemonic) not in [12, 15, 18, 21, 24]:
                raise ValueError(
                    "Mnemonic has invalid size. This should be 12,15,18,21 or 24 words"
                )
            keypair = Keypair.create_from_mnemonic(
                " ".join(mnemonic), ss58_format=basedai.__ss58_format__
            )
            if not suppress:
                display_mnemonic_msg(keypair, "personalkey")
        elif seed is not None:
            keypair = Keypair.create_from_seed(
                seed, ss58_format=basedai.__ss58_format__
            )
        else:
            # json is not None
            if (
                not isinstance(json, tuple)
                or len(json) != 2
                or not isinstance(json[0], (str, dict))
                or not isinstance(json[1], str)
            ):
                raise ValueError(
                    "json must be a tuple of (json_data: str | Dict, passphrase: str)"
                )

            json_data, passphrase = json
            keypair = Keypair.create_from_encrypted_json(
                json_data, passphrase, ss58_format=basedai.__ss58_format__
            )

        self.set_personalkey(keypair, encrypt=use_password, overwrite=overwrite)
        self.set_personalkeypub(keypair, overwrite=overwrite)
        return self

    # Short name for regenerate_personalkey
    regen_personalkey = regenerate_personalkey

    @overload
    def regenerate_computekey(
        self,
        mnemonic: Optional[Union[list, str]] = None,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        ...

    @overload
    def regenerate_computekey(
        self,
        seed: Optional[str] = None,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        ...

    @overload
    def regenerate_computekey(
        self,
        json: Optional[Tuple[Union[str, Dict], str]] = None,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> "wallet":
        ...

    def regenerate_computekey(
        self,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
        **kwargs,
    ) -> "wallet":
        """Regenerates the computekey from passed mnemonic or seed, encrypts it with the user's password and saves the file.

        Args:
            mnemonic: (Union[list, str], optional):
                Key mnemonic as list of words or string space separated words.
            seed: (str, optional):
                Seed as hex string.
            json: (Tuple[Union[str, Dict], str], optional):
                Restore from encrypted JSON backup as ``(json_data: Union[str, Dict], passphrase: str)``.
            use_password (bool, optional):
                Is the created key password protected.
            overwrite (bool, optional):
                Determies if this operation overwrites the computekey under the same path ``<wallet path>/<wallet name>/computekeys/<computekey>``.
        Returns:
            wallet (basedai.wallet):
                This object with newly created computekey.
        """
        if len(kwargs) == 0:
            raise ValueError("Must pass either mnemonic, seed, or json")

        # Get from kwargs
        mnemonic = kwargs.get("mnemonic", None)
        seed = kwargs.get("seed", None)
        json = kwargs.get("json", None)

        if mnemonic is None and seed is None and json is None:
            raise ValueError("Must pass either mnemonic, seed, or json")
        if mnemonic is not None:
            if isinstance(mnemonic, str):
                mnemonic = mnemonic.split()
            if len(mnemonic) not in [12, 15, 18, 21, 24]:
                raise ValueError(
                    "Mnemonic has invalid size. This should be 12,15,18,21 or 24 words"
                )
            keypair = Keypair.create_from_mnemonic(
                " ".join(mnemonic), ss58_format=basedai.__ss58_format__
            )
            if not suppress:
                display_mnemonic_msg(keypair, "computekey")
        elif seed is not None:
            keypair = Keypair.create_from_seed(
                seed, ss58_format=basedai.__ss58_format__
            )
        else:
            # json is not None
            if (
                not isinstance(json, tuple)
                or len(json) != 2
                or not isinstance(json[0], (str, dict))
                or not isinstance(json[1], str)
            ):
                raise ValueError(
                    "json must be a tuple of (json_data: str | Dict, passphrase: str)"
                )

            json_data, passphrase = json
            keypair = Keypair.create_from_encrypted_json(
                json_data, passphrase, ss58_format=basedai.__ss58_format__
            )

        self.set_computekey(keypair, encrypt=use_password, overwrite=overwrite)
        return self

    # Short name for regenerate_computekey
    regen_computekey = regenerate_computekey
