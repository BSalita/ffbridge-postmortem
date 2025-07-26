# FFBridge Library Usage Guide

This guide provides comprehensive documentation for the FFBridge Postmortem Analyzer - a toolkit for extracting, analyzing, and performing post-mortem analysis of bridge tournament data from the French Bridge Federation (FFBridge) system.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Type Definitions](#type-definitions)
4. [API Integration](#api-integration)
5. [Data Processing](#data-processing)
6. [Enhanced HTML Parsing](#enhanced-html-parsing)
7. [Streamlit Interface](#streamlit-interface)
8. [Authentication](#authentication)
9. [Error Handling](#error-handling)
10. [Examples](#examples)

## Overview

The FFBridge Postmortem Analyzer consists of several key components:

- **Data Extraction**: Automated extraction from FFBridge and BridgePlus websites
- **API Integration**: RESTful API access with bearer token authentication
- **Data Processing**: Convert raw tournament data into structured formats
- **Interactive Analysis**: Streamlit-based web interface for post-mortem analysis
- **Type Safety**: Comprehensive type definitions for better code quality
- **Enhanced HTML Parsing**: Robust Unicode-aware parsing with French-to-English translation

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for web automation)
playwright install chromium

# Set up environment variables
cp env_template.txt .env
# Edit .env with your FFBridge credentials

# Run type checking
mypy ffbridge_streamlit.py
```

### Basic Usage

```python
import asyncio
from mlBridgeLib.mlBridgeBPLib import get_all_boards_for_team_async

async def extract_tournament_data():
    # Extract all boards for a team
    result = await get_all_boards_for_team_async(
        tr="S202602",  # Tournament ID
        cl="5802079",  # Club ID
        sc="A",        # Section
        eq="212"       # Team number
    )
    
    boards_df = result['boards']
    frequency_df = result['score_frequency']
    
    print(f"Extracted {len(boards_df)} boards")
    print(f"Opponent info: {boards_df['Opponent_Pair_Names'][0]} vs {boards_df['Team_Name'][0]}")
    return result

# Run extraction
data = asyncio.run(extract_tournament_data())
```

### Interactive Analysis

```bash
# Launch Streamlit interface
streamlit run ffbridge_streamlit.py
```

## Type Definitions

The application uses comprehensive type definitions for better code quality and IDE support:

### ApiUrlConfig

```python
class ApiUrlConfig(TypedDict):
    url: str
    should_cache: bool
```

Defines the structure for API endpoint configuration:
- `url`: Complete API endpoint URL
- `should_cache`: Boolean flag for response caching

### ApiUrlsDict

```python
class ApiUrlsDict(TypedDict):
    simultaneous_deals: ApiUrlConfig
    simultaneous_description_by_organization_id: ApiUrlConfig
    simultaneous_tournaments_by_organization_id: ApiUrlConfig
    my_infos: ApiUrlConfig
    members: ApiUrlConfig
    person: ApiUrlConfig
    organization_by_person_organization_id: ApiUrlConfig
    person_by_person_organization_id: ApiUrlConfig
```

Defines the complete API configuration structure with all supported endpoints.

### DataFramesDict

```python
class DataFramesDict(TypedDict):
    boards: Optional[pl.DataFrame]
    score_frequency: Optional[pl.DataFrame]
```

Defines the structure for processed data with Polars DataFrames.

## API Integration

### Authentication

```python
def make_api_request_licencie(full_url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Make authenticated API request to FFBridge licencie endpoints"""
    default_headers = {
        'Authorization': f'Bearer {st.session_state.ffbridge_bearer_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    if headers:
        default_headers.update(headers)
    
    try:
        response = requests.get(full_url, headers=default_headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None
```

### Supported Endpoints

The library supports various FFBridge API endpoints:

- `simultaneous_deals`: Extract deal information for simultaneous tournaments
- `simultaneous_description_by_organization_id`: Get tournament descriptions
- `simultaneous_tournaments_by_organization_id`: List tournaments by organization
- `my_infos`: Get current user information
- `members`: Search and retrieve member information
- `person`: Get detailed person information
- `organization_by_person_organization_id`: Get organization details
- `person_by_person_organization_id`: Get person details by organization

## Data Processing

### DataFrame Operations

```python
def ShowDataFrameTable(df: pl.DataFrame, key: str, query: str = 'SELECT * FROM self', show_sql_query: bool = True) -> Optional[pl.DataFrame]:
    """Display and query a Polars DataFrame with SQL interface"""
    if df is None or len(df) == 0:
        st.warning(f"No data available for {key}")
        return None
    
    # Execute SQL query on DataFrame
    result_df = df.query(query)
    
    # Display results
    st.dataframe(result_df, use_container_width=True)
    
    if show_sql_query:
        st.code(query, language='sql')
    
    return result_df
```

### Data Augmentation

```python
def augment_df(df: pl.DataFrame) -> pl.DataFrame:
    """Augment bridge data with calculated statistics and analysis"""
    if df is None or len(df) == 0:
        return df
    
    # Create augmentation object
    augmenter = mlBridgeAugmentLib.mlBridgeAugment(df)
    
    # Perform augmentations
    augmenter.augment()
    
    return augmenter.df
```

## Enhanced HTML Parsing

### Unicode-Aware Parsing

The library includes robust HTML parsing capabilities with comprehensive French-to-English translation:

```python
# Comprehensive strain mapping including Unicode symbols
FRENCH_TO_ENGLISH_STRAIN_MAP = {
    'P': 'S',    # Spades (Piques)
    'C': 'H',    # Hearts (Cœurs)
    'K': 'D',    # Diamonds (Carreau)
    'T': 'C',    # Clubs (Trèfle)
    'sa': 'N',   # No Trump (Sans Atout) - lowercase
    'SA': 'N',   # No Trump (Sans Atout) - uppercase
    '♠': 'S',    # Unicode spade symbol
    '♥': 'H',    # Unicode heart symbol
    '♦': 'D',    # Unicode diamond symbol
    '♣': 'C',    # Unicode club symbol
    'pique': 'S',    # French class names
    'coeur': 'H',    # French class names
    'carreau': 'D',  # French class names
    'trefle': 'C',   # French class names
}
```

### Contract Extraction

Enhanced contract extraction with robust regex patterns:

```python
# Extract contract information from HTML
contract_span_pattern = r'<span\s+[^>]*class=["\'][^"\']*gros[^"\']*["\'][^>]*>.*?Contrat\s.*?<\/span>'
contract_span_match = re.search(contract_span_pattern, page_content, re.DOTALL)

if contract_span_match:
    contract_span_html = contract_span_match.group(0)
    
    # Extract declarer
    declarer_match = re.search(r'\(([NESW])\)', contract_span_html)
    if declarer_match:
        declarer = translate_direction(declarer_match.group(1))
    
    # Extract level
    level_match = re.search(r':\s*(\d+)', contract_span_html)
    if level_match:
        level = level_match.group(1)
    
    # Extract suit with enhanced regex
    suit_span_match = re.search(
        r'Contrat.*?(?:<span class="(pique|coeur|carreau|trefle)">[^<]*</span>|(SA))', 
        contract_span_html
    )
    if suit_span_match and suit_span_match.group(1):  # Suit class found
        suit_class = suit_span_match.group(1)
    elif suit_span_match and suit_span_match.group(2):  # SA match found
        suit_class = suit_span_match.group(2)
    else:
        raise ValueError(f"Could not find suit in contract span: {contract_span_html}")
    
    strain = FRENCH_TO_ENGLISH_STRAIN_MAP[suit_class]
    contract = level + strain + declarer
```

### Vulnerability Translation

Robust vulnerability translation with error handling:

```python
FRENCH_TO_ENGLISH_VULNERABILITY_MAP = {
    'Personne': 'None',
    'Nord-Sud': 'N_S',
    'Est-Ouest': 'E_W',
    'Tous': 'Both',
}

# Extract vulnerability with error handling
vul_match = re.search(r'Vulnérabilité[^:]*:\s*([^<]+)', page_content)
if vul_match:
    french_vul = vul_match.group(1).strip()
    if french_vul:
        # Use direct mapping with error handling
        vul = FRENCH_TO_ENGLISH_VULNERABILITY_MAP[french_vul]
        logger.info(f"Extracted vulnerability: {vul}")
    else:
        vul = 'None'
```

### Card Distribution Extraction

Enhanced card extraction with Unicode support:

```python
async def extract_cards_from_page_async(page) -> List[str]:
    """Extract card distributions from HTML with Unicode support"""
    card_sequences = []
    
    # Find all card divs with enhanced selector
    card_divs = await page.query_selector_all('div.flex-grow-1.ms-3.gros')
    
    for div in card_divs:
        card_text = await div.text_content()
        if card_text and card_text.strip():
            # Handle Unicode symbols and French notation
            card_text = card_text.strip()
            card_sequences.append(card_text)
    
    return card_sequences
```

### Error Handling in HTML Parsing

```python
# Graduated error handling with detailed diagnostics
if len(route_data) < 5:
    logger.warning(f"Very few route records extracted: only {len(route_data)} records found")
    logger.warning("This might indicate:")
    logger.warning("  1. Team played few boards")
    logger.warning("  2. HTML parsing issues")
    logger.warning("  3. Page structure changed")
    logger.warning("  4. Unicode encoding problems")
```

## Streamlit Interface

### Session State Management

The application uses Streamlit's session state for managing application state:

```python
def initialize_session_state() -> None:
    """Initialize Streamlit session state variables"""
    first_time_defaults = {
        'single_dummy_sample_count': 10,
        'debug_mode': False,
        'show_sql_query': True,
        'use_historical_data': False,
        'do_not_cache_df': True,
        'con': duckdb.connect(),
        'cache_dir': 'cache',
    }
```

### UI Components

#### Sidebar Interface

```python
def create_sidebar() -> None:
    """Create the main sidebar interface"""
    st.sidebar.text_input(
        "Enter ffbridge player license number", 
        on_change=player_search_input_on_change, 
        placeholder=st.session_state.player_license_number, 
        key='player_search_input'
    )
```

#### Chat Interface

```python
def chat_input_on_submit() -> None:
    """Handle chat input submission and process SQL queries"""
    prompt = st.session_state.main_prompt_chat_input
    sql_query = process_prompt_macros(prompt)
    # Process and display results
```

### Report Generation

```python
def write_report() -> None:
    """Write and display the bridge game analysis report"""
    report_title = f"Bridge Game Postmortem Report"
    report_person = f"Personalized for {st.session_state.player_name} ({st.session_state.player_license_number})"
    # Generate comprehensive analysis report
```

## Authentication

### Token Management

```python
def initialize_ffbridge_bearer_token() -> None:
    """Initialize FFBridge Bearer token from .env file or environment variables"""
    load_dotenv()
    
    # Load Lancelot token
    token = os.getenv('FFBRIDGE_BEARER_TOKEN_LANCELOT')
    if token:
        st.session_state.ffbridge_bearer_token = token
    
    # Load EASI token
    token = os.getenv('FFBRIDGE_EASI_TOKEN')
    if token:
        st.session_state.ffbridge_easi_token = token
```

### Automated Authentication

For automated token extraction, use the Playwright-based authentication:

```python
from ffbridge_auth_playwright import get_bearer_token_playwright_sync

# Extract token using browser automation
token = get_bearer_token_playwright_sync()
```

## Error Handling

### API Error Handling

```python
def make_api_request_licencie(full_url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(full_url, headers=default_headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
```

### HTML Parsing Error Handling

```python
# Comprehensive error handling for HTML parsing
try:
    # Extract data from HTML
    data = await extract_data_from_html(page)
except ValueError as e:
    logger.error(f"Data extraction failed: {e}")
    # Provide fallback or default values
except Exception as e:
    logger.error(f"Unexpected error during HTML parsing: {e}")
    # Log detailed diagnostic information
```

### Data Validation

```python
def validate_dataframe_structure(df: pl.DataFrame, expected_columns: List[str]) -> bool:
    """Validate DataFrame structure and data types"""
    if df is None:
        return False
    
    # Check required columns
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check data types and values
    for col in df.columns:
        if df[col].null_count() == len(df):
            st.warning(f"Column {col} contains only null values")
    
    return True
```

## Examples

### Tournament Analysis

```python
async def analyze_tournament():
    """Complete tournament analysis example"""
    # Extract tournament data
    result = await get_all_boards_for_team_async(
        tr="S202602", cl="5802079", sc="A", eq="212"
    )
    
    # Process and augment data
    boards_df = result['boards']
    augmented_df = augment_df(boards_df)
    
    # Generate analysis
    ShowDataFrameTable(augmented_df, "tournament_analysis")
    
    return augmented_df
```

### API Data Extraction

```python
def extract_api_data():
    """Extract data using FFBridge API"""
    # Configure API endpoints
    api_urls = {
        'simultaneous_deals': {
            'url': 'https://api.ffbridge.fr/api/v1/simultaneous-deals',
            'should_cache': True
        }
    }
    
    # Make API request
    data = make_api_request_licencie(api_urls['simultaneous_deals']['url'])
    
    # Process response
    if data:
        df = pl.DataFrame(data)
        return df
    
    return None
```

### Custom SQL Queries

```python
def custom_analysis():
    """Perform custom analysis using SQL queries"""
    # Load data
    df = load_bridge_data()
    
    # Custom SQL analysis
    query = """
    SELECT 
        Board,
        Contract,
        Score,
        AVG(Score) OVER (PARTITION BY Board) as AvgScore,
        Score - AVG(Score) OVER (PARTITION BY Board) as ScoreDiff
    FROM self
    WHERE Score IS NOT NULL
    ORDER BY Board, Score DESC
    """
    
    result = ShowDataFrameTable(df, "custom_analysis", query)
    return result
```

## Troubleshooting

### Common Issues

1. **Unicode Parsing Errors**: Ensure proper encoding handling for French characters
2. **HTML Structure Changes**: Update regex patterns if website structure changes
3. **API Authentication**: Verify bearer tokens are valid and not expired
4. **Data Type Mismatches**: Check DataFrame schemas and column types

### Debug Mode

Enable debug mode for detailed logging:

```python
# Set debug mode in session state
st.session_state.debug_mode = True

# Debug information will be displayed in the interface
if st.session_state.debug_mode:
    st.write("Debug information:", debug_data)
```

## Performance Optimization

### Caching Strategies

```python
# Enable caching for API responses
api_urls = {
    'endpoint': {
        'url': 'https://api.example.com/data',
        'should_cache': True
    }
}
```

### Parallel Processing

```python
# Use async functions for concurrent operations
async def process_multiple_boards():
    tasks = []
    for board_num in range(1, 37):
        task = get_board_by_number_async(tr, cl, board_num)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## Contributing

### Code Style

- Use type hints for all function parameters and return values
- Follow async naming convention (`_async` suffix for async functions)
- Include comprehensive error handling
- Add docstrings for all public functions

### Testing

```bash
# Run comprehensive test suite
python test_bridge_library.py

# Run type checking
mypy ffbridge_streamlit.py

# Run specific tests
python test_type_definitions.py
```

### Documentation

- Update this guide when adding new features
- Include code examples for new functionality
- Document any breaking changes
- Maintain type definitions for new data structures 