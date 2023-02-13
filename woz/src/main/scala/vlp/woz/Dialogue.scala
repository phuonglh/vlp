package vlp.woz

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.catalyst.ScalaReflection
val scalaSchema = ScalaReflection.schemaFor[SlotValues].dataType.asInstanceOf[StructType]

final case class Slot(
  copy_from: String,
  exclusive_end: Long,
  slot: String,
  start: Long,
  value: String
)

final case class SlotValues()

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

final case class Dialogue(
  dialogue_id: String,
  services: Array[String],
  turns: Array[Turn]
)
