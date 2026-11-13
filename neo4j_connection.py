"""
============================================================
CONSTRUCTION CENTRALISEE DU DRIVER NEO4J
============================================================
Sur cette machine (Windows), le driver Neo4j valide par defaut le certificat
TLS via le magasin de certificats systeme (ssl.SSLContext.load_default_certs(),
cf. neo4j.TrustSystemCAs -- le defaut de neo4j+s://). Constate en pratique le
2026-11-13 : ca echoue avec "SSLCertVerificationError: self-signed certificate
in certificate chain" sur l'instance Aura du projet, MEME si la racine
concernee (SSL.com, utilisee par Aura) est bien presente et valide dans le
magasin Windows (bug connu de l'enumeration de certificats Windows par le
module ssl de CPython) -- une connexion TCP/TLS brute vers le meme hote
reussit sans probleme. Forcer la validation via certifi (deja une dependance
du projet, cf. requirements.txt) resout le probleme immediatement, sans
affaiblir la verification : c'est toujours une vraie chaine de confiance,
juste une source de certificats racine differente.
"""
from neo4j import GraphDatabase, TrustCustomCAs
import certifi

# neo4j+s/bolt+s -> equivalent non chiffre + chiffrement/confiance passes
# explicitement : trusted_certificates ne peut pas se combiner avec un schema
# qui definit deja lui-meme le chiffrement (ConfigurationError sinon).
_ENCRYPTED_SCHEMES = {"neo4j+s": "neo4j", "bolt+s": "bolt"}


def build_neo4j_driver(uri: str, username: str, password: str):
    """Equivalent de GraphDatabase.driver(uri, auth=(username, password)),
    qui valide le certificat TLS via certifi plutot que le magasin systeme."""
    scheme, _, rest = uri.partition("://")
    plain_scheme = _ENCRYPTED_SCHEMES.get(scheme)
    if plain_scheme is None:
        # Deja non chiffre (tests locaux) ou deja gere autrement (+ssc) :
        # rien a adapter.
        return GraphDatabase.driver(uri, auth=(username, password))

    return GraphDatabase.driver(
        f"{plain_scheme}://{rest}",
        auth=(username, password),
        encrypted=True,
        trusted_certificates=TrustCustomCAs(certifi.where()),
    )
