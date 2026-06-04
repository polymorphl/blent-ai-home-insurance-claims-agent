"""Simulated claim declaration examples for end-to-end pipeline testing."""

from datetime import date, timedelta


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


_today = date.today()
_yesterday = _fmt(_today - timedelta(days=1))
_2d_ago = _fmt(_today - timedelta(days=2))
_3d_ago = _fmt(_today - timedelta(days=3))
_10d_ago = _fmt(_today - timedelta(days=10))
_30d_ago = _fmt(_today - timedelta(days=30))


EXAMPLES = [
    # --- Approved cases ---
    {
        "id": "ex1_water_damage_approved",
        "description": "Water damage — relative date ('hier soir'), real photo, 2-turn",
        "turns": [
            (
                "Bonjour,\n\nIl y a eu une fuite dans ma cuisine hier soir à cause de mon voisin du dessus. "
                "Son lave-vaisselle a été mal installé et du coup, le mur est infiltré d'eau et la peinture "
                "se détache (ci-joint une photo).\n\nCordialement.\n\n[Pièce jointe : WaterDamage_100.jpg]"
            ),
            f"La fuite a été constatée le {_yesterday}.",
        ],
    },
    {
        "id": "ex2_fire_approved",
        "description": "Fire — complete in 1 turn, real fire photos",
        "turns": [
            (
                f"Bonjour,\n\nLe {_2d_ago}, un feu s'est déclaré dans la chambre à cause d'un appareil "
                "défectueux, et a endommagé une grande partie de la pièce. Je souhaiterai être indemnisé "
                "pour pouvoir effectuer les travaux nécessaires.\n\nBien cordialement.\n\n"
                "[Pièce jointe : FireDamage_45.jpg]\n[Pièce jointe : FireDamage_31.jpg]"
            ),
        ],
    },
    {
        "id": "ex3_theft_no_photos",
        "description": "Theft — missing date initially; no photos available → rejected at conformity (photos required for theft)",
        "turns": [
            (
                "Bonjour, on m'a cambriolé ce matin, les voleurs sont passés par le vélux de la chambre "
                "et ont volé tous les appareils électroniques. Merci de me contacter rapidement."
            ),
            f"Le cambriolage a eu lieu le {_yesterday}. Je n'avais pas de photos de l'effraction.",
        ],
    },
    # --- Rejected cases ---
    {
        "id": "ex4_theft_deadline_exceeded",
        "description": "Theft — declaration 30 days after incident, exceeds 2-day limit",
        "turns": [
            (
                f"Bonjour, j'ai été victime d'un cambriolage le {_30d_ago}. Les voleurs ont forcé la porte "
                "d'entrée et emporté ma télévision, mon ordinateur et des bijoux. "
                "Je dépose ma déclaration aujourd'hui.\n\n[Pièce jointe : photo_effraction.jpg]"
            ),
        ],
    },
    {
        "id": "ex5_water_damage_no_photos",
        "description": "Water damage — no photos, rejected at conformity check",
        "turns": [
            (
                f"Bonjour,\n\nUne fuite est apparue dans mon salon le {_3d_ago} suite à une rupture de "
                "canalisation. Les dégâts sont importants : parquet gondolé, cloison humide. "
                "Je n'ai malheureusement pas pu prendre de photos.\n\nCordialement."
            ),
        ],
    },
    {
        "id": "ex6_fire_wrong_photos",
        "description": "Fire — recent date, but photos show water damage → rejected for photo incoherence",
        "turns": [
            (
                f"Bonjour,\n\nUn incendie s'est déclaré le {_2d_ago} dans ma cuisine. "
                "Les dommages sont importants.\n\nBien cordialement.\n\n"
                "[Pièce jointe : WaterDamage_176.jpg]\n[Pièce jointe : WaterDamage_100.jpg]"
            ),
        ],
    },
]
