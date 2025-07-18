# ffbridge-postmortem

FFBridge Postmortem Analyzer - Bridge Tournament Data Extraction and Analysis

## Overview

This project provides a comprehensive toolkit for extracting, analyzing, and post-mortem analysis of bridge tournament data from the French Bridge Federation (FFBridge) system. It includes web scraping capabilities, data processing tools, and interactive analysis interfaces.

## Key Features

- **Web Scraping**: Automated extraction of bridge tournament data from FFBridge and BridgePlus websites
- **Data Processing**: Convert raw tournament data into structured formats for analysis
- **Interactive Analysis**: Streamlit-based web interface for post-mortem game analysis
- **Authentication**: Automated bearer token management for FFBridge API access
- **Bridge Library**: Comprehensive library for bridge deal analysis and augmentation

## Main Components

### 1. Data Extraction (`mlBridgeLib/mlBridgeBPLib.py`)
- Tournament discovery and club extraction
- Team and board results scraping
- Individual board/deal extraction with PBN format
- Parallel processing for high-performance data extraction
- All async functions follow `_async` naming convention
- Enhanced HTML parsing with dynamic board number extraction
- Robust opponent information extraction using specific HTML structure

### 2. Interactive Analysis (`ffbridge_streamlit.py`)
- Web-based interface for game analysis
- Chat-based postmortem analysis with AI assistance
- Support for both simultaneous and RRN tournaments
- Real-time data visualization and statistics

### 3. Authentication (`ffbridge_auth_playwright.py`)
- Automated bearer token extraction using Playwright
- Environment variable management for credentials
- Support for both Lancelot and EASI token types

### 4. Bridge Analysis Library (`mlBridgeLib/`)
- Deal analysis and augmentation tools
- Double dummy analysis integration
- Bidding and play analysis
- Machine learning utilities for bridge AI

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set up environment variables
cp env_template.txt .env
# Edit .env with your FFBridge credentials

# Optional: Run type checking
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

## Documentation

- **[Library Usage Guide](LIBRARY_USAGE_GUIDE.md)**: Comprehensive guide for using the bridge scraping library
- **[Authentication Guide](old/AUTOMATION_USAGE.md)**: Setup and usage of automated authentication
- **Test Suite**: Run `python test_bridge_library.py` for comprehensive testing
- **Type Checking**: Run `mypy ffbridge_streamlit.py` for static type analysis

## Recent Updates

### Version 2.3.0 (Latest)
- **Comprehensive Type Hints**: Added full type annotations throughout the codebase for better IDE support and code quality
- **Enhanced Documentation**: Improved function docstrings with detailed parameter and return type descriptions
- **Type Safety**: Added TypedDict definitions for complex data structures and API configurations
- **Development Tools**: Added mypy and typing-extensions for static type checking
- **Code Quality**: Better organized requirements.txt with version constraints and categorized dependencies

### Version 2.2.0 (Previous)
- **Enhanced HTML Parsing**: Dynamic board number extraction from HTML instead of hardcoded assumptions
- **Improved Opponent Extraction**: Uses specific HTML div structure (`<div class="col">contre <span class="paires">...`) for reliable opponent information
- **Better Error Handling**: Graduated warnings instead of hard failures, with detailed diagnostic information
- **Data Correspondence**: Board numbers, scores, and matchpoints are extracted from same HTML rows to ensure accuracy
- **Robust Validation**: Enhanced data validation with detailed troubleshooting information
- **Fixed Board Parsing**: Removed hardcoded board skip patterns (19, 20) that were tournament-specific

### Version 2.1.0 (Previous)
- **New Data Fields**: Added `Opponent_Pair_Section` field for better opponent tracking
- **Enhanced Validation**: Automatic team existence checking with detailed diagnostics
- **Improved Error Handling**: Better error messages and graceful handling of missing data
- **Section Validation**: Automatic detection of opponent section mismatches

### Version 2.0.0 (Previous)
- **Async Naming Convention**: All async functions now use `_async` suffix
- **New Functions**: Added `get_board_for_player_async()`, `get_board_for_team()`, and optimized player-specific extraction
- **Enhanced Error Handling**: Improved robustness in web scraping operations
- **Authentication Improvements**: Better token management and environment variable handling

## Technical Improvements

### Enhanced Data Extraction
- **Dynamic Board Discovery**: Automatically detects board numbers from HTML content
- **Opponent Information**: Comprehensive opponent tracking with direction, names, number, and section
- **Data Integrity**: Maintains correspondence between board numbers, scores, and matchpoints
- **Error Diagnostics**: Detailed troubleshooting information for common issues

### Improved HTML Parsing
```python
# OLD: Hardcoded assumptions
for i in range(1, 27):
    if i not in [19, 20]:  # Skip boards 19 and 20 (not played)
        board_nums.append(i)

# NEW: Dynamic extraction from HTML
board_data = []
data_rows = await page.query_selector_all('div.row')
for row in data_rows:
    # Extract board number, score, and matchpoints from same row
    # Ensures correspondence between related data
```

### Robust Opponent Extraction
```python
# Uses specific HTML structure for reliable extraction
opponent_div = await page.query_selector('div.col:has-text("contre")')
if opponent_div:
    paires_span = await opponent_div.query_selector('span.paires')
    if paires_span:
        # Parse opponent text like "NS : SININGE - KRIEF (113 A)"
        paires_text = paires_span.text_content()
        # Extract direction, names, number, and section
```

## Project Structure

```
ffbridge-postmortem/
├── mlBridgeLib/           # Core bridge analysis library
│   ├── mlBridgeBPLib.py   # BridgePlus scraping library (main)
│   ├── mlBridgeLib.py     # Core bridge utilities
│   ├── mlBridgeAugmentLib.py  # Deal augmentation tools
│   └── ...                # Other bridge analysis modules
├── streamlitlib/          # Streamlit utilities
├── cache/                 # Cached API responses
├── competitions/          # Tournament data storage
├── results/               # Analysis results
├── old/                   # Legacy code and documentation
├── api-*.bat             # API testing scripts
├── ffbridge_streamlit.py  # Main Streamlit application
├── test_bridge_library.py # Comprehensive test suite
├── LIBRARY_USAGE_GUIDE.md # Detailed usage documentation
└── requirements.txt       # Python dependencies
```

## Data Schema

### Board Results DataFrame
- `Board`, `Score`, `Percentage`, `Contract`, `Declarer`, `Lead`
- `Pair_Names`, `Pair_Direction`, `Pair_Number`, `Section`
- `Opponent_Pair_Names`, `Opponent_Pair_Direction`, `Opponent_Pair_Number`, `Opponent_Pair_Section`

### Boards DataFrame
- `Board`, `PBN`, `Top`, `Dealer`, `Vul`, `Contract`, `Result`, `Score`
- `Team_Name`, `Pair_Number`, `Section`, `Tournament_ID`, `Club_ID`
- `Opponent_Pair_Names`, `Opponent_Pair_Direction`, `Opponent_Pair_Number`, `Opponent_Pair_Section`

### Score Frequency DataFrame
- `Board`, `Score`, `Frequency`, `Matchpoints_NS`, `Matchpoints_EW`

## Performance Features

- **Parallel Processing**: Concurrent extraction of multiple boards
- **Shared Browser Context**: Reuse browser instances for multiple requests
- **Intelligent Waiting**: Adaptive delays based on page load times
- **Robust Error Recovery**: Graceful handling of network issues and page changes
- **Efficient Data Structures**: Polars DataFrames for high-performance data operations

## Code Quality

- **Type Safety**: Comprehensive type hints throughout the codebase using Python's typing system
- **Static Analysis**: Support for mypy static type checking
- **Documentation**: Detailed docstrings with parameter and return type descriptions
- **Structured Data**: TypedDict definitions for complex API configurations and data structures
- **IDE Support**: Enhanced autocomplete and error detection in modern IDEs

## Error Handling

The library provides comprehensive error handling with detailed diagnostics:

```python
# Enhanced error messages with suggestions
if len(route_data) < 5:
    logger.warning(f"Very few route records extracted: only {len(route_data)} records found")
    logger.warning("This might indicate:")
    logger.warning("  1. Team played few boards")
    logger.warning("  2. HTML parsing issues")
    logger.warning("  3. Page structure changed")
```

## Contributing

This project is actively developed and maintained. Key areas for contribution:
- Additional bridge analysis algorithms
- UI/UX improvements for the Streamlit interface
- Performance optimizations for data extraction
- Support for additional tournament formats
- Enhanced error handling and diagnostics
- Type safety improvements and additional type hints
- Documentation enhancements

## Testing

Run the comprehensive test suite:

```bash
python test_bridge_library.py
```

This tests all major functions and provides detailed performance metrics.

### Type Checking

For static type analysis:

```bash
# Check all main files
python check_types.py

# Check specific file
mypy ffbridge_streamlit.py

# Check with custom configuration
mypy ffbridge_streamlit.py --config-file mypy.ini
```

This will check for type errors and provide suggestions for improving type safety.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the [Library Usage Guide](LIBRARY_USAGE_GUIDE.md) for detailed documentation
2. Run the test suite to verify installation
3. Review error messages for diagnostic information
4. Check the `old/` directory for legacy documentation
