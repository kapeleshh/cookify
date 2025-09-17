"""
NLP Processor - Processes transcribed text to extract cooking-related information
"""

import os
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Class for processing transcribed text to extract cooking-related information.
    """
    
    def __init__(self, model_name="en_core_web_lg", use_custom_ner=True, entity_confidence=0.7):
        """
        Initialize the NLPProcessor.
        
        Args:
            model_name (str, optional): spaCy model to use. Defaults to "en_core_web_lg".
            use_custom_ner (bool, optional): Whether to use custom NER for cooking entities. Defaults to True.
            entity_confidence (float, optional): Minimum confidence for entity recognition. Defaults to 0.7.
        """
        self.model_name = model_name
        self.use_custom_ner = use_custom_ner
        self.entity_confidence = entity_confidence
        self.nlp = None
        
        # Lazy load the model when needed
    
    def _load_model(self):
        """
        Load the spaCy model.
        """
        if self.nlp is not None:
            return
        
        try:
            import spacy
            
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            
            # Add custom components if configured
            if self.use_custom_ner:
                self._add_custom_components()
            
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
    
    def _add_custom_components(self):
        """
        Add custom components to the spaCy pipeline for cooking-specific NLP.
        """
        # This is a placeholder for adding custom components
        # In a real implementation, we would add custom components here
        logger.info("Adding custom components to spaCy pipeline")
    
    def process(self, transcription):
        """
        Process transcribed text to extract cooking-related information.
        
        Args:
            transcription (dict): Transcription result with text and segments.
            
        Returns:
            dict: Extracted cooking-related information.
        """
        self._load_model()
        
        text = transcription.get("text", "")
        segments = transcription.get("segments", [])
        
        logger.info(f"Processing transcription: {len(text)} characters, {len(segments)} segments")
        
        try:
            # Process full text
            doc = self.nlp(text)
            
            # Extract ingredients
            ingredients = self._extract_ingredients(doc)
            
            # Extract quantities
            quantities = self._extract_quantities(doc)
            
            # Extract cooking actions
            actions = self._extract_cooking_actions(doc)
            
            # Extract cooking tools
            tools = self._extract_cooking_tools(doc)
            
            # Extract cooking times
            times = self._extract_cooking_times(doc)
            
            # Extract temperatures
            temperatures = self._extract_temperatures(doc)
            
            # Extract title
            title = self._extract_title(doc, segments)
            
            # Extract servings
            servings = self._extract_servings(doc)
            
            # Create structured result
            result = {
                "title": title,
                "servings": servings,
                "ingredients": ingredients,
                "quantities": quantities,
                "actions": actions,
                "tools": tools,
                "times": times,
                "temperatures": temperatures
            }
            
            logger.info(f"Extracted {len(ingredients)} ingredients, {len(actions)} actions, {len(tools)} tools")
            return result
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            # Return empty result as fallback
            return {
                "title": "",
                "servings": "",
                "ingredients": [],
                "quantities": [],
                "actions": [],
                "tools": [],
                "times": [],
                "temperatures": []
            }
    
    def _extract_ingredients(self, doc):
        """
        Extract ingredients from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of ingredients with name, quantity, and unit.
        """
        ingredients = []
        
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
        # Common food entities in spaCy
        food_entities = ["FOOD", "PRODUCT", "SUBSTANCE"]
        
        # Extract entities that might be ingredients
        for ent in doc.ents:
            if ent.label_ in food_entities or self._is_likely_ingredient(ent.text):
                # Try to find associated quantity and unit
                quantity, unit = self._find_quantity_and_unit(ent)
                
                ingredient = {
                    "name": ent.text,
                    "quantity": quantity,
                    "unit": unit
                }
                
                ingredients.append(ingredient)
        
        # Deduplicate ingredients
        unique_ingredients = []
        seen_names = set()
        
        for ingredient in ingredients:
            name = ingredient["name"].lower()
            if name not in seen_names:
                seen_names.add(name)
                unique_ingredients.append(ingredient)
        
        return unique_ingredients
    
    def _is_likely_ingredient(self, text):
        """
        Check if text is likely an ingredient.
        
        Args:
            text (str): Text to check.
            
        Returns:
            bool: True if likely an ingredient, False otherwise.
        """
        # Common ingredients
        common_ingredients = [
            "salt", "pepper", "sugar", "flour", "butter", "oil", "water", "milk",
            "egg", "garlic", "onion", "tomato", "potato", "carrot", "chicken",
            "beef", "pork", "fish", "rice", "pasta", "cheese", "cream", "yogurt",
            "vinegar", "lemon", "lime", "orange", "apple", "banana", "berry",
            "chocolate", "vanilla", "cinnamon", "oregano", "basil", "thyme",
            "rosemary", "parsley", "cilantro", "ginger", "soy sauce", "honey",
            "maple syrup", "mustard", "ketchup", "mayonnaise", "bread", "tortilla"
        ]
        
        text_lower = text.lower()
        
        # Check if text contains a common ingredient
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                return True
        
        return False
    
    def _find_quantity_and_unit(self, entity):
        """
        Find quantity and unit associated with an ingredient entity.
        
        Args:
            entity (spacy.tokens.Span): Entity span.
            
        Returns:
            tuple: (quantity, unit)
        """
        # This is a simplified implementation
        # In a real implementation, we would use dependency parsing and more sophisticated techniques
        
        # Common cooking units
        cooking_units = [
            "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
            "gram", "g", "kg", "ml", "l", "liter", "pinch", "dash", "handful"
        ]
        
        # Look for numbers and units before the entity
        quantity = ""
        unit = ""
        
        # Check previous tokens
        for i in range(1, 4):  # Look up to 3 tokens before
            if entity.start - i >= 0:
                prev_token = entity.doc[entity.start - i]
                
                # Check if token is a number
                if prev_token.like_num:
                    quantity = prev_token.text
                
                # Check if token is a unit
                if prev_token.text.lower() in cooking_units:
                    unit = prev_token.text
        
        return quantity, unit
    
    def _extract_quantities(self, doc):
        """
        Extract quantities from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of quantities with value, unit, and position.
        """
        quantities = []
        
        # Common cooking units
        cooking_units = [
            "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
            "gram", "g", "kg", "ml", "l", "liter", "pinch", "dash", "handful"
        ]
        
        # Extract quantities using pattern matching
        for i, token in enumerate(doc):
            if token.like_num:
                # Check if next token is a unit
                if i + 1 < len(doc) and doc[i + 1].text.lower() in cooking_units:
                    quantity = {
                        "value": token.text,
                        "unit": doc[i + 1].text,
                        "start": token.idx,
                        "end": doc[i + 1].idx + len(doc[i + 1].text)
                    }
                    
                    quantities.append(quantity)
        
        return quantities
    
    def _extract_cooking_actions(self, doc):
        """
        Extract cooking actions from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of cooking actions with action and position.
        """
        actions = []
        
        # Common cooking verbs
        cooking_verbs = [
            "add", "bake", "beat", "blend", "boil", "break", "bring", "brown",
            "chop", "combine", "cook", "cool", "cover", "cut", "dice", "drain",
            "drizzle", "drop", "dry", "fill", "flip", "fold", "fry", "garnish",
            "grate", "grill", "heat", "knead", "layer", "marinate", "mash", "melt",
            "mix", "pour", "preheat", "prepare", "press", "reduce", "remove", "rinse",
            "roast", "roll", "rub", "season", "serve", "set", "simmer", "slice",
            "spread", "sprinkle", "stir", "strain", "stuff", "taste", "toss", "transfer",
            "turn", "whip", "whisk"
        ]
        
        # Extract cooking verbs
        for token in doc:
            if token.lemma_.lower() in cooking_verbs:
                # Get the surrounding context
                context_start = max(0, token.i - 5)
                context_end = min(len(doc), token.i + 6)
                context = doc[context_start:context_end].text
                
                action = {
                    "action": token.lemma_,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "context": context
                }
                
                actions.append(action)
        
        return actions
    
    def _extract_cooking_tools(self, doc):
        """
        Extract cooking tools from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of cooking tools with tool and position.
        """
        tools = []
        
        # Common cooking tools
        cooking_tools = [
            "bowl", "pan", "pot", "skillet", "knife", "spoon", "fork", "whisk",
            "spatula", "blender", "mixer", "grater", "peeler", "cutting board",
            "measuring cup", "measuring spoon", "oven", "stove", "microwave",
            "refrigerator", "freezer", "grill", "griddle", "slow cooker",
            "pressure cooker", "food processor", "colander", "strainer"
        ]
        
        # Extract cooking tools using pattern matching
        for tool in cooking_tools:
            # Check if tool is a single word or multiple words
            if " " in tool:
                # Multi-word tool
                if tool.lower() in doc.text.lower():
                    # Find all occurrences
                    for match in re.finditer(tool, doc.text, re.IGNORECASE):
                        tool_obj = {
                            "tool": tool,
                            "start": match.start(),
                            "end": match.end()
                        }
                        
                        tools.append(tool_obj)
            else:
                # Single-word tool
                for token in doc:
                    if token.text.lower() == tool.lower():
                        tool_obj = {
                            "tool": token.text,
                            "start": token.idx,
                            "end": token.idx + len(token.text)
                        }
                        
                        tools.append(tool_obj)
        
        return tools
    
    def _extract_cooking_times(self, doc):
        """
        Extract cooking times from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of cooking times with value, unit, and position.
        """
        times = []
        
        # Time units
        time_units = ["minute", "min", "hour", "hr", "second", "sec"]
        
        # Extract times using pattern matching
        for i, token in enumerate(doc):
            if token.like_num:
                # Check if next token is a time unit
                if i + 1 < len(doc) and any(unit in doc[i + 1].text.lower() for unit in time_units):
                    time_obj = {
                        "value": token.text,
                        "unit": doc[i + 1].text,
                        "start": token.idx,
                        "end": doc[i + 1].idx + len(doc[i + 1].text)
                    }
                    
                    times.append(time_obj)
        
        return times
    
    def _extract_temperatures(self, doc):
        """
        Extract temperatures from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of temperatures with value, unit, and position.
        """
        temperatures = []
        
        # Temperature units
        temp_units = ["degree", "degrees", "°", "°C", "°F", "celsius", "fahrenheit"]
        
        # Extract temperatures using pattern matching
        for i, token in enumerate(doc):
            if token.like_num:
                # Check if next token is a temperature unit
                if i + 1 < len(doc) and any(unit in doc[i + 1].text.lower() for unit in temp_units):
                    temp_obj = {
                        "value": token.text,
                        "unit": doc[i + 1].text,
                        "start": token.idx,
                        "end": doc[i + 1].idx + len(doc[i + 1].text)
                    }
                    
                    temperatures.append(temp_obj)
        
        return temperatures
    
    def _extract_title(self, doc, segments):
        """
        Extract recipe title from a spaCy doc and segments.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            segments (list): List of transcription segments.
            
        Returns:
            str: Recipe title.
        """
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
        # Check first few segments for title
        if segments:
            first_segment = segments[0]["text"]
            
            # Common title patterns
            title_patterns = [
                r"(?:making|preparing|cooking|how to make|recipe for|today we're making)\s+(.+)",
                r"(?:welcome to|today's recipe is|today we're going to make)\s+(.+)",
                r"(?:this is|here's|let's make)\s+(.+)"
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, first_segment, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Fallback: Use first noun phrase
        for chunk in doc.noun_chunks:
            return chunk.text
        
        return ""
    
    def _extract_servings(self, doc):
        """
        Extract servings information from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            str: Servings information.
        """
        # Common serving patterns
        serving_patterns = [
            r"(?:serves|servings|makes|yields|enough for)\s+(\d+)",
            r"(?:recipe for|feeds)\s+(\d+)",
            r"(\d+)\s+(?:servings|portions|people)"
        ]
        
        for pattern in serving_patterns:
            match = re.search(pattern, doc.text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
