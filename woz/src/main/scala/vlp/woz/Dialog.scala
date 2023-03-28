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
  `attraction-area`: Array[String],
  `attraction-name`: Array[String],
  `attraction-type`: Array[String],
  // `hospital-department`: Array[String], // only in dev split
  `hotel-area`: Array[String],
  `hotel-bookday`: Array[String],
  `hotel-bookpeople`: Array[String],
  `hotel-bookstay`: Array[String],
  `hotel-internet`: Array[String],
  `hotel-name`: Array[String],
  `hotel-parking`: Array[String],
  `hotel-pricerange`: Array[String],
  `hotel-stars`: Array[String],
  `hotel-type`: Array[String],
  `restaurant-area`: Array[String],
  `restaurant-bookday`: Array[String],
  `restaurant-bookpeople`: Array[String],
  `restaurant-booktime`: Array[String],
  `restaurant-food`: Array[String],
  `restaurant-name`: Array[String],
  `restaurant-pricerange`: Array[String],
  `taxi-arriveby`: Array[String],
  `taxi-departure`: Array[String],
  `taxi-destination`: Array[String],
  `taxi-leaveat`: Array[String],
  `train-arriveby`: Array[String],
  `train-bookpeople`: Array[String],
  `train-day`: Array[String],
  `train-departure`: Array[String],
  `train-destination`: Array[String],
  `train-leaveat`: Array[String]
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
