from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from pathlib import Path

p = Path(__file__).parent.parent / ".ssh"
p.mkdir(parents=True, exist_ok=True)
priv = p / "fake_checker_key"
pub = p / "fake_checker_key.pub"
key = ed25519.Ed25519PrivateKey.generate()
priv_bytes = key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
pub_key = key.public_key()
pub_bytes = pub_key.public_bytes(
    encoding=serialization.Encoding.OpenSSH,
    format=serialization.PublicFormat.OpenSSH,
)
priv.write_bytes(priv_bytes)
pub.write_bytes(pub_bytes)
print(pub_bytes.decode())
