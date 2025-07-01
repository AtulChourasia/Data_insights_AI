import re
from typing import Dict, List, Set
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NLQueryExtractor:
    def __init__(self):
        # Download required NLTK data (run once)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # SQL operation keywords
        self.select_keywords = {
            'show', 'display', 'get', 'find', 'list', 'retrieve', 'fetch', 
            'give', 'return', 'see', 'view', 'what', 'which', 'tell'
        }
        
        self.aggregation_keywords = {
            'count': ['count', 'number', 'how many', 'total number'],
            'sum': ['sum', 'total', 'add up', 'sum up'],
            'avg': ['average', 'mean', 'avg'],
            'max': ['maximum', 'max', 'highest', 'largest', 'most', 'top'],
            'min': ['minimum', 'min', 'lowest', 'smallest', 'least', 'bottom']
        }
        
        self.condition_keywords = {
            'where': ['where', 'with', 'having', 'that have', 'who have', 'which have'],
            'greater': ['greater than', 'more than', 'above', 'over', 'higher than'],
            'less': ['less than', 'below', 'under', 'lower than', 'fewer than'],
            'equal': ['equal to', 'equals', 'is', 'are'],
            'like': ['contains', 'includes', 'has', 'with', 'like'],
            'between': ['between', 'from', 'to', 'range']
        }
        
        self.join_keywords = {
            'and', 'with', 'along with', 'together with', 'including', 
            'related to', 'associated with', 'connected to'
        }
        
        self.sort_keywords = {
            'order': ['order by', 'sort by', 'arrange by', 'sorted', 'ordered'],
            'desc': ['descending', 'desc', 'highest first', 'largest first', 'decreasing'],
            'asc': ['ascending', 'asc', 'lowest first', 'smallest first', 'increasing']
        }
        
        self.limit_keywords = {
            'top', 'first', 'limit', 'only', 'just'
        }

    def extract_nl_elements(self, user_query: str, available_tables: List[str] = None, 
                           available_columns: List[str] = None) -> Dict[str, List[str]]:
        """
        Extract SQL-relevant elements from natural language user query
        
        Args:
            user_query: Natural language question from user
            available_tables: List of table names in the database
            available_columns: List of column names across all tables
            
        Returns:
            Dictionary with extracted elements
        """
        if not user_query:
            return self._empty_elements()
        
        # Clean and tokenize query
        query_lower = user_query.lower().strip()
        tokens = word_tokenize(query_lower)
        
        # Initialize elements dictionary
        elements = {
            'select_type': [],      # What type of selection (show, count, etc.)
            'columns': [],          # Column names mentioned
            'tables': [],           # Table names mentioned  
            'conditions': [],       # WHERE conditions
            'aggregations': [],     # COUNT, SUM, AVG, etc.
            'joins': [],           # JOIN indicators
            'sorting': [],         # ORDER BY indicators
            'limits': [],          # LIMIT indicators
            'operators': []        # Comparison operators
        }
        
        # Extract different types of elements
        elements['select_type'] = self._extract_select_type(query_lower, tokens)
        elements['aggregations'] = self._extract_aggregations(query_lower)
        elements['conditions'] = self._extract_conditions(query_lower)
        elements['sorting'] = self._extract_sorting(query_lower)
        elements['limits'] = self._extract_limits(query_lower, tokens)
        elements['operators'] = self._extract_operators(query_lower)
        elements['joins'] = self._extract_joins(query_lower)
        
        # Extract tables and columns using available schema
        if available_tables:
            elements['tables'] = self._extract_tables(query_lower, available_tables)
        if available_columns:
            elements['columns'] = self._extract_columns(query_lower, available_columns)
            
        # If no schema provided, extract potential table/column names
        if not available_tables and not available_columns:
            potential_entities = self._extract_potential_entities(tokens)
            elements['tables'] = potential_entities['tables']
            elements['columns'] = potential_entities['columns']
        
        return elements
    
    def _empty_elements(self) -> Dict[str, List[str]]:
        """Return empty elements dictionary"""
        return {
            'select_type': [],
            'columns': [], 
            'tables': [],
            'conditions': [],
            'aggregations': [],
            'joins': [],
            'sorting': [],
            'limits': [],
            'operators': []
        }
    
    def _extract_select_type(self, query: str, tokens: List[str]) -> List[str]:
        """Extract what type of SELECT operation is needed"""
        select_types = []
        
        for keyword in self.select_keywords:
            if keyword in query:
                select_types.append(keyword)
        
        return list(set(select_types))
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation functions needed"""
        aggregations = []
        
        for agg_type, keywords in self.aggregation_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    aggregations.append(agg_type)
                    break
        
        return list(set(aggregations))
    
    def _extract_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions and filtering requirements"""
        conditions = []
        
        # Look for condition patterns
        condition_patterns = [
            r'where\s+(.+?)(?:\s+order|\s+group|$)',
            r'with\s+(.+?)(?:\s+order|\s+group|$)',
            r'having\s+(.+?)(?:\s+order|\s+group|$)',
            r'that\s+have\s+(.+?)(?:\s+order|\s+group|$)',
            r'who\s+have\s+(.+?)(?:\s+order|\s+group|$)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, query)
            conditions.extend(matches)
        
        return [cond.strip() for cond in conditions if cond.strip()]
    
    def _extract_sorting(self, query: str) -> List[str]:
        """Extract ORDER BY requirements"""
        sorting = []
        
        for sort_type, keywords in self.sort_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    sorting.append(sort_type)
                    break
        
        return list(set(sorting))
    
    def _extract_limits(self, query: str, tokens: List[str]) -> List[str]:
        """Extract LIMIT requirements and numbers"""
        limits = []
        
        # Look for limit keywords followed by numbers
        for i, token in enumerate(tokens):
            if token in self.limit_keywords:
                # Look for number after limit keyword
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limits.append(f"{token}_{tokens[i + 1]}")
                else:
                    limits.append(token)
        
        # Extract standalone numbers that might indicate limits
        numbers = re.findall(r'\b(\d+)\b', query)
        for num in numbers:
            if any(keyword in query for keyword in self.limit_keywords):
                limits.append(f"limit_{num}")
        
        return list(set(limits))
    
    def _extract_operators(self, query: str) -> List[str]:
        """Extract comparison operators"""
        operators = []
        
        for op_type, keywords in self.condition_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    operators.append(op_type)
                    break
        
        return list(set(operators))
    
    def _extract_joins(self, query: str) -> List[str]:
        """Extract JOIN requirements"""
        joins = []
        
        for keyword in self.join_keywords:
            if keyword in query:
                joins.append(keyword)
        
        return list(set(joins))
    
    def _extract_tables(self, query: str, available_tables: List[str]) -> List[str]:
        """Extract table names from query using available schema"""
        found_tables = []
        
        for table in available_tables:
            # Check exact match and variations
            table_variations = [
                table.lower(),
                table.lower().replace('_', ' '),
                table.lower().replace('_', ''),
                self.lemmatizer.lemmatize(table.lower()),
                # Check plural/singular forms
                table.lower() + 's' if not table.lower().endswith('s') else table.lower()[:-1]
            ]
            
            for variation in table_variations:
                if variation in query:
                    found_tables.append(table)
                    break
        
        return list(set(found_tables))
    
    def _extract_columns(self, query: str, available_columns: List[str]) -> List[str]:
        """Extract column names from query using available schema"""
        found_columns = []
        
        for column in available_columns:
            # Check exact match and variations
            column_variations = [
                column.lower(),
                column.lower().replace('_', ' '), 
                column.lower().replace('_', ''),
                self.lemmatizer.lemmatize(column.lower()),
                # Common column name patterns
                column.lower().replace('id', '').replace('name', '').strip('_')
            ]
            
            for variation in column_variations:
                if variation in query and len(variation) > 2:  # Avoid very short matches
                    found_columns.append(column)
                    break
        
        return list(set(found_columns))
    
    def _extract_potential_entities(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Extract potential table/column names when schema is not available"""
        # Filter out stop words and common words
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and 
            token.isalpha() and 
            len(token) > 2 and
            token not in {'show', 'get', 'find', 'from', 'with', 'have', 'that', 'who', 'which'}
        ]
        
        # Capitalize first letter (common for table names)
        potential_tables = [token.capitalize() for token in filtered_tokens if len(token) > 4]
        potential_columns = [token.lower() for token in filtered_tokens]
        
        return {
            'tables': potential_tables[:3],  # Limit to most likely candidates
            'columns': potential_columns[:5]
        }

# Usage example:
def example_usage():
    extractor = NLQueryExtractor()
    
    # Example with schema
    available_tables = ['Employee', 'Project', 'Department', 'Assignment']
    available_columns = ['name', 'employee_id', 'project_name', 'salary', 'department_name', 'status']
    
    user_query = "show all names from Employee table who have some projects"
    
    elements = extractor.extract_nl_elements(
        user_query, 
        available_tables=available_tables,
        available_columns=available_columns
    )
    
    

