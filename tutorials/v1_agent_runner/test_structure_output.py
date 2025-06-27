from openai import OpenAI
from pydantic import BaseModel

from adalflow.utils import setup_env

setup_env()

client = OpenAI()


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


schema = CalendarEvent.model_json_schema()
print(schema)

# {'properties': {'name': {'title': 'Name', 'type': 'string'}, 'date': {'title': 'Date', 'type': 'string'}, 'participants': {'items': {'type': 'string'}, 'title': 'Participants', 'type': 'array'}}, 'required': ['name', 'date', 'participants'], 'title': 'CalendarEvent', 'type': 'object'}


# response = client.chat.completions.create(
#     model="gpt-4o-2024-08-06",
#     messages=[
#         {"role": "system", "content": "Extract the event information."},
#         {
#             "role": "user",
#             "content": "Alice and Bob are going to a science fair on Friday.",
#         },
#     ],
#     response_format={
#         "type": "json_schema",
#         "json_schema": {
#             "name": "calendar_event",
#             "schema": schema,
#             "strict": True,  # reject missing / extra keys
#         },
#     },
# )
# print(response)

# response = client.responses.parse(
#     model="gpt-4o-2024-08-06",
#     input=[
#         {"role": "system", "content": "Extract the event information."},
#         {
#             "role": "user",
#             "content": "Alice and Bob are going to a science fair on Friday.",
#         },
#     ],
#     text_format=CalendarEvent,
# )
# print(response)

# event = response.output_parsed
# print(f"Event: {event}")
