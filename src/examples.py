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
        "description": "Water damage — relative date ('hier soir'), photo present, 2-turn",
        "turns": [
            (
                "Bonjour,\n\nIl y a eu une fuite dans ma cuisine hier soir à cause de mon voisin du dessus. "
                "Son lave-vaisselle a été mal installé et du coup, le mur est infiltré d'eau et la peinture "
                "se détache (ci-joint une photo).\n\nCordialement.\n\n[Pièce jointe : IMG_4580.jpg]"
            ),
            f"La fuite a été constatée le {_yesterday}.",
        ],
    },
    {
        "id": "ex2_fire_approved",
        "description": "Fire — complete in 1 turn, explicit recent date",
        "turns": [
            (
                f"Bonjour,\n\nLe {_2d_ago}, un feu s'est déclaré dans la chambre à cause d'un appareil "
                "défectueux, et a endommagé une grande partie de la pièce. Je souhaiterai être indemnisé "
                "pour pouvoir effectuer les travaux nécessaires.\n\nBien cordialement.\n\n"
                "[Pièce jointe : Chambre_1.jpg]\n[Pièce jointe : Chambre_2.jpg]"
            ),
        ],
    },
    {
        "id": "ex3_theft_approved",
        "description": "Theft — missing date and photos initially, provided on turn 2",
        "turns": [
            (
                "Bonjour, on m'a cambriolé ce matin, les voleurs sont passés par le vélux de la chambre "
                "et ont volé tous les appareils électroniques. Merci de me contacter rapidement."
            ),
            f"Le cambriolage a eu lieu le {_yesterday}.\n[Pièce jointe : photo_effraction.jpg]",
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
                "Je dépose ma déclaration aujourd'hui.\n\n[Pièce jointe : porte_forcee.jpg]"
            ),
        ],
    },
    {
        "id": "ex5_water_damage_no_photos",
        "description": "Water damage — no photos, rejected at validation",
        "turns": [
            (
                f"Bonjour,\n\nUne fuite est apparue dans mon salon le {_3d_ago} suite à une rupture de "
                "canalisation. Les dégâts sont importants : parquet gondolé, cloison humide. "
                "Je n'ai malheureusement pas pu prendre de photos.\n\nCordialement."
            ),
        ],
    },
    {
        "id": "ex6_fire_old_date",
        "description": "Fire — declaration 10 days after incident, exceeds 5-day limit",
        "turns": [
            (
                f"Bonjour,\n\nUn incendie s'est déclaré le {_10d_ago} dans ma cuisine à cause d'une plaque "
                "de cuisson laissée allumée. Les dommages sont conséquents. Je reviens de vacances et "
                "je déclare le sinistre dès maintenant.\n\nBien cordialement.\n\n"
                "[Pièce jointe : cuisine_incendie.jpg]"
            ),
        ],
    },
]
