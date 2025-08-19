from dataclasses import dataclass, field
from typing import List, Optional
import adalflow as adal
from adalflow.core import DataClass, required_field
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient

from adalflow.components.output_parsers.outputs import JsonOutputParser, JsonOutputParserPydanticModel
from adalflow.utils import setup_env

# OpenAI's structured output approach
from openai import OpenAI
from pydantic import BaseModel

setup_env()

# ===============================
# SHARED DATA MODELS
# ===============================

# Pydantic models for OpenAI structured output
class Participants(BaseModel):
    names: List[str]
    addresses: List[str]

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: Participants

# AdalFlow DataClass models
@dataclass
class ParticipantsData(DataClass):
    names: List[str] = field(
        metadata={"desc": "List of participant names"},
        default_factory=list
    )
    addresses: Optional[List[str]] = field(
        metadata={"desc": "List of participant addresses"},
        default_factory=list
    )

@dataclass
class CalendarEventData(DataClass):
    name: str = field(
        metadata={"desc": "Name of the calendar event"},
        default_factory=required_field()
    )
    date: str = field(
        metadata={"desc": "Date of the event"},
        default_factory=required_field()
    )
    participants: ParticipantsData = field(
        metadata={"desc": "Event participants information"},
        default_factory=required_field()
    )

# ===============================
# OPENAI STRUCTURED OUTPUT
# ===============================

def test_openai_structured_output():
    """Test OpenAI's native structured output parsing."""
    print("\n=== Testing OpenAI Structured Output ===")
    
    client = OpenAI()
    
    try:
        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "Extract the event information. Use a synthesic, very complciated string as the address for the particpant to test very complicated json parsing"},
                {
                    "role": "user",
                    "content": "Alice and Bob are going to a science fair on Friday.",
                },
            ],
            text_format=CalendarEvent,
        )

        event = response.output_parsed
        print(f"OpenAI Response: {event}")
        
        return event
        
    except Exception as e:
        print(f"OpenAI structured output failed: {e}")
        return None

# ===============================
# ADALFLOW GENERATOR + JSON PARSER
# ===============================

class AdalFlowEventExtractor(adal.Component):
    """AdalFlow component using Generator with JsonOutputParser."""
    
    def __init__(self, model_client: adal.ModelClient, model_kwargs: dict):
        super().__init__()
        
        # Set up output parser
        self.output_parser = JsonOutputParser(
            data_class=CalendarEventData,
            return_data_class=True
        )
        self.output_parser_pydantic = JsonOutputParserPydanticModel(
            pydantic_model=CalendarEvent,
            return_pydantic_object=True
        )
        # Template for the Generator
        self.template = r"""
<START_OF_SYSTEM_MESSAGE>
{{system_prompt}}

{{output_format_str}}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
{{user_input}}
<END_OF_USER_MESSAGE>
        """.strip()
        
        # Set up Generator
        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=self.template,
            output_processors=self.output_parser,
            use_cache=False,
        )
    
    def call(self, user_input: str, system_prompt: str) -> adal.GeneratorOutput:
        """Extract event information using AdalFlow Generator + JsonOutputParser."""
        # print(f" output_format_str: {self.output_parser.format_instructions()}")
        prompt_kwargs = {
            "system_prompt": system_prompt,
            "output_format_str": self.output_parser.format_instructions(),
            "user_input": user_input
        }
        
        return self.llm(prompt_kwargs=prompt_kwargs)

def test_adalflow_json_parser():
    """Test AdalFlow's Generator with JsonOutputParser."""
    print("\n=== Testing AdalFlow Generator + JsonOutputParser ===")
    
    model_config = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 500,
        }
    }
    
    extractor = AdalFlowEventExtractor(
        model_client=model_config["model_client"],
        model_kwargs=model_config["model_kwargs"]
    )
    
    system_prompt = "Extract the event information. g"
    user_input = "Alice and Bob are going to a science fair on Friday."
    
    try:
        response = extractor.call(user_input, system_prompt)
        
        # print(f"AdalFlow Response: {response}")
        
        if response.data:
            event_data = response.data
            print(f"Event Name: {event_data.name}")
            print(f"Date: {event_data.date}")
            print(f"Participants: {event_data.participants.names}")
            print(f"Addresses: {event_data.participants.addresses}")
        else:
            print(f"Failed to parse, raw response: {response.raw_response}")
            
        return response
        
    except Exception as e:
        print(f"AdalFlow extraction failed: {e}")
        return None

# ===============================
# COMPARISON AND BENCHMARKING
# ===============================

def compare_approaches():
    """Compare both approaches side by side."""
    print("\n" + "="*60)
    print("STRUCTURED OUTPUT COMPARISON")
    print("="*60)
    
    # Test cases with increasingly complex scenarios
    test_cases = [
        "Alice and Bob are going to a science fair on Friday.",
        "John, Mary, and David will attend the birthday party on December 25th.",
        "The team meeting with Sarah, Mike, Tom, and Lisa is scheduled for next Monday at the conference room.",
        # Complex address challenge
        "Dr. Katherine Martinez-O'Sullivan and Prof. Ahmed bin Rashid Al-Maktoum will attend the International Conference on AI Ethics on March 15th, 2024. Katherine lives at 1247-B Château de Malmaison, Apt. #47/C, Neuilly-sur-Seine, Île-de-France 92200, France (GPS: 48.8738°N, 2.1667°E) and Ahmed resides at Building 47/Tower C/Floor 23/Unit 2301-A, Sheikh Zayed Road Complex, Near Dubai Mall Metro Station, Dubai, United Arab Emirates, P.O. Box 112233-ABUDHABI-UAE (Emergency Contact: +971-4-XXX-YYYY).",
        # JSON-breaking characters and edge cases
        "Meeting participants: Alex \"The Coder\" Johnson (address: 123 Main St., Unit #5-B\nSecond Floor\n\"Special Building\"\nCity: Austin\"Texas\" 78701\nCountry: USA\"America\"), Maria José Rodríguez-Pérez (Calle de José Ortega y Gasset, 29\n28006 Madrid\nEspaña), and Zhang Wei 张伟 (北京市朝阳区建国门外大街1号\nBeijing 100001\n中华人民共和国). Event: Tech Summit 2024 on February 29th.",
        # Unicode, emojis, and special formatting
        "🎉 Grand Opening Party 🎊 on Saturday, Jan 20th! Attendees: Σωκράτης Παπαδόπουλος (Πλατεία Συντάγματος 1, Αθήνα 10563, Ελλάδα 🇬🇷), Владимир Иванович Петров (Красная площадь, дом 1, Москва 109012, Российская Федерация 🇷🇺), and السيد أحمد محمد علي (شارع التحرير رقم 15، القاهرة 11511، جمهورية مصر العربية 🇪🇬).",
        # Nested quotes and complex punctuation
        "Annual \"Best Practices\" Workshop featuring: Robert \"Bob\" O'Malley-Smith Jr., Ph.D., M.D. (address: \"The Heights\", Building A-1, Suite 100-C, Floor 2.5, 987 Oak Tree Lane & Maple Street Intersection, South Hampton, NY 11968-1234 [Note: Use side entrance during construction]), Dr. Mary-Catherine Van Der Berg-Williams III (Château \"Les Trois Rosés\", 45 Rue de la Paix & Boulevard Saint-Germain, 7ème Arrondissement, Paris 75007, République Française [Buzzer code: \"2024-SECRET\"]), scheduled for December 31st, 2024 at 11:59 PM.",
        # JSON structure breakers and extreme formatting
        "Emergency meeting tomorrow! Participants: {\"name\": \"John Smith\", \"role\": \"CEO\"} living at [Address Object]: {\"street\": \"123 {Main} Street\", \"apt\": \"#[456-B]\", \"city\": \"New {York}\", \"state\": \"NY\", \"zip\": \"[10001-2345]\", \"coordinates\": {\"lat\": 40.7589, \"lng\": -73.9851}, \"special_notes\": \"Ring bell 3x, say 'pizza delivery', wait 30sec, then ring 2x more\"}, and Jane Doe at \"[CLASSIFIED LOCATION]\" {GPS: [REDACTED], Building: [UNKNOWN], Floor: [N/A]}"
    ]
    adalflow_count = 0 
    openai_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case}")
        
        # Test OpenAI
        openai_result = test_openai_structured_output_simple(test_case)
        
        # Test AdalFlow
        adalflow_result = test_adalflow_json_parser_simple(test_case)

        print("adalflow_result:", adalflow_result.data)
        print("openai_result:", openai_result)


        
        # Compare results
        print("\n--- Comparison ---")
        if openai_result and isinstance(openai_result, CalendarEvent):
            print("✅Openai")
            openai_count += 1
        if adalflow_result and adalflow_result.data and (isinstance(adalflow_result.data, CalendarEvent) or isinstance(adalflow_result.data, CalendarEventData)):
            print("✅AdalFlow")
            adalflow_count += 1


    print(f"OpenAI Count: {openai_count}, AdalFlow Count: {adalflow_count}")

def test_openai_structured_output_simple(user_input: str):
    """Simplified OpenAI test for comparison."""
    client = OpenAI()
    
    try:
        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "Extract the event information."},
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            text_format=CalendarEvent,
        )

        event = response.output_parsed
        return event
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def test_adalflow_json_parser_simple(user_input: str):
    """Simplified AdalFlow test for comparison."""
 
 

    model_config={
        "model_client": AnthropicAPIClient(),
        "model_kwargs": {
            "model": "claude-sonnet-4-20250514"
        }
    }
    model_config={
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-4o-mini",
        }
    }
    
    extractor = AdalFlowEventExtractor(
        model_client=model_config["model_client"],
        model_kwargs=model_config["model_kwargs"]
    )
    
    system_prompt = "Extract the event information. "
    
    try:
        return extractor.call(user_input, system_prompt)
    except Exception as e:
        print(f"AdalFlow error: {e}")
        return None

if __name__ == "__main__":
    # Run individual tests
    openai_result = test_openai_structured_output()
    adalflow_result = test_adalflow_json_parser()
    
    # Run comparison
    compare_approaches()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("OpenAI Structured Output:")
    print("  + Native JSON schema validation")
    print("  + Guaranteed structure compliance")
    print("  - Requires specific OpenAI models")
    print("  - Less flexibility in processing pipeline")
    
    print("\nAdalFlow Generator + JsonOutputParser:")
    print("  + Model-agnostic approach")
    print("  + Flexible processing pipeline")
    print("  + Integration with optimization framework")
    print("  - May require retry logic for parsing failures")
    print("  + Better integration with AdalFlow ecosystem")