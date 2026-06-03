EXTRACT_SYSTEM_PROMPT = (
    "You are an insurance claim data extractor. "
    "Analyze the policyholder's message and call the extract_claim_fields function "
    "with whatever information you can identify. Always call the function, even if most fields are null.\n"
    "Today's date is {today}.\n"
    "Rules:\n"
    "- For date: use ISO format YYYY-MM-DD. Messages are in French — dates like '10/09/2025' follow DD/MM/YYYY format. "
    "Resolve relative dates using today's date (e.g. 'hier', 'ce matin'). Return null only if the date is truly unresolvable.\n"
    "- For incident_type: map to water_damage (leak, infiltration, flooding), "
    "fire (fire, explosion, smoke), or theft (burglary, robbery, vandalism).\n"
    "- For has_photos: true if attachments are listed in the message, false if the user explicitly says they have no photos, null if not mentioned."
)

FOLLOW_UP_SYSTEM_PROMPT = (
    "You are a helpful insurance claims assistant for AssurHabitat. "
    "You are collecting information to process a home insurance claim. "
    "Be polite and concise. Ask only for the missing information. "
    "Do not repeat questions about information already provided. "
    "Respond in French."
)

_FIELD_LABELS = {
    "date": "la date de survenance du sinistre",
    "incident_type": "la nature du sinistre (dégât des eaux, incendie ou cambriolage)",
    "description": "une description des dommages constatés",
    "has_photos": "des photos des dommages en pièce jointe",
}


def follow_up_prompt(claim_data: dict, missing_fields: list[str]) -> str:
    """Generate a French prompt for the model to request missing claim information."""
    known = {k: v for k, v in claim_data.items() if v is not None}
    missing_labels = [_FIELD_LABELS[f] for f in missing_fields]
    return (
        f"Informations déjà collectées : {known}\n"
        f"Informations manquantes : {', '.join(missing_labels)}\n"
        "Rédigez un message demandant poliment ces informations à l'assuré."
    )
