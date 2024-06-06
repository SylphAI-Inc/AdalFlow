from lightrag.core.string_parser import (
    JsonParser,
)

from lightrag.core.functional import (
    extract_json_str,
    fix_json_missing_commas,
    fix_json_escaped_single_quotes,
)

import pytest


##################################################
# Test cases for extract_json_str function
##################################################
def test_extract_json_str_valid():
    text = '{"name": "John", "age": 30}'
    assert extract_json_str(text) == '{"name": "John", "age": 30}'


def test_extract_json_str_with_missing_brace():
    text = '{"name": "John", "age": 30'
    assert (
        extract_json_str(text, add_missing_right_brace=True)
        == '{"name": "John", "age": 30}'
    )


def test_extract_json_str_no_json():
    text = "No JSON here"
    with pytest.raises(ValueError):
        extract_json_str(text)


##################################################
# Test cases for fix_json_formatting function
##################################################
def test_fix_json_formatting_adds_commas():
    # Malformed JSON string missing commas
    input_json = '{"name": "John" "age": 30 "attributes": {"height": 180 "weight": 70}}'
    expected_json = (
        '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}}'
    )

    result_json = fix_json_missing_commas(input_json)

    # Assert to check if the commas were added correctly
    assert (
        result_json == expected_json
    ), f"Expected: {expected_json}, but got: {result_json}"


def test_fix_json_formatting_no_change_needed():
    correct_json = (
        '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}}'
    )
    result_json = fix_json_missing_commas(correct_json)

    # Assert to check that no changes are made to the already correct JSON
    assert result_json == correct_json, f"Expected no change, but got: {result_json}"


##################################################
# Test cases for fix_json_escaped_single_quotes function
##################################################
def test_fix_json_escaped_single_quotes():
    input_json = r"""
        {
        "thought": "I need to get the answer from llm_tool and finish the task.",
        "action": "finish('Li\'s pet Apple is 7 years and 2 months old.')"
        }
        """
    expected_json = r"""
        {
        "thought": "I need to get the answer from llm_tool and finish the task.",
        "action": "finish('Li's pet Apple is 7 years and 2 months old.')"
        }
        """
    result_json = fix_json_escaped_single_quotes(input_json)
    print(f"result_json: {result_json}")
    assert result_json == expected_json


##################################################
# Test cases for class JsonParser
##################################################
def test_json_parser_valid():
    parser = JsonParser()
    text = '{"name": "John", "age": 30}'
    result = parser(text)
    assert result == {"name": "John", "age": 30}


def test_json_parser_fix_missing_brace():
    parser = JsonParser()
    text = '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}'
    result = parser(text)
    assert result == {
        "name": "John",
        "age": 30,
        "attributes": {"height": 180, "weight": 70},
    }


def test_json_parser_fix_missing_commas():
    parser = JsonParser()
    text = '{"name": "John" "age": 30 "attributes": {"height": 180 "weight": 70}}'
    result = parser(text)
    assert result == {
        "name": "John",
        "age": 30,
        "attributes": {"height": 180, "weight": 70},
    }


def test_json_parser_handling_decode_error():
    parser = JsonParser()
    # Deliberately malformed JSON that is also problematic for YAML
    text = '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}]}'
    with pytest.raises(ValueError) as excinfo:
        parser(text)
    assert "Got invalid JSON object" in str(excinfo.value)


def test_json_parser_escape_single_quotes():
    # we did not quote double quotes in the JSON string, so it is invalid
    parser = JsonParser()
    text = r"""
    {
    "thought": "The template 2 has been fetched and shown to the founder. I should ask for the specific company information and founder's profile to personalize the email.",
    "action": "ask_for_information("company information and founder\'s profile")"
    }
    """
    with pytest.raises(ValueError) as excinfo:
        result = parser(text)
        print(f"result: {result}")
    assert "Got invalid JSON object" in str(excinfo.value)
