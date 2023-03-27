package vlp.woz

final case class Slot(
  copy_from: String,
  exclusive_end: Option[Long],
  slot: String,
  start: Option[Long],
  value: String
)

// final case class SlotValues() // empty for dialogue act experiments
final case class SlotValues(  
)

final case class State(
  active_intent: String,
  requested_slots: Array[String],
  slot_values: SlotValues
)

final case class Frame(
  actions: Array[String],
  service: String,
  slots: Array[Slot],
  state: State
)

final case class Turn(
  frames: Array[Frame],
  speaker: String,
  turn_id: String,
  utterance: String
)

final case class Dialog(
  dialogue_id: String,
  services: Array[String],
  turns: Array[Turn]
)
