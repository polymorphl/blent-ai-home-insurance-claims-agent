"""Simulated claim declaration examples for testing the declaration agent."""

EXAMPLES = [
    {
        "id": "ex1_water_damage",
        "turns": [
            (
                "Bonjour,\n\nIl y a eu une fuite dans ma cuisine hier soir à cause de mon voisin du dessus. "
                "Son lave-vaisselle a été mal installé et du coup, le mur est infiltré d'eau et la peinture "
                "se détache (ci-joint une photo).\n\nCordialement.\n\n[Pièce jointe : IMG_4580.jpg]"
            ),
            "La fuite a été constatée le 15/03/2025.",
        ],
    },
    {
        "id": "ex2_theft",
        "turns": [
            (
                "Bonjour, on m'a cambriolé ce matin, les voleurs sont passés par le vélux de la chambre "
                "et ont volé tous les appareils électroniques. Merci de me contacter rapidement."
            ),
            "Le cambriolage a eu lieu le 10/09/2025.\n[Pièce jointe : photo_effraction.jpg]",
        ],
    },
    {
        "id": "ex3_fire_complete",
        "turns": [
            (
                "Bonjour,\n\nLe 10/09/2025, un feu s'est déclaré dans la chambre à cause d'un appareil "
                "défectueux, et a endommagé une grande partie de la pièce. Je souhaiterai être indemnisé "
                "pour pouvoir effectuer les travaux nécessaires.\n\nBien cordialement.\n\n"
                "[Pièce jointe : Chambre_1.jpg]\n[Pièce jointe : Chambre_2.jpg]"
            ),
        ],
    },
]
