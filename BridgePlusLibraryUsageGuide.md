# Bridge Library Usage Guide

This guide shows how to use the BridgePlus Bridge Scraper Library for extracting bridge tournament data. This replaces the `demo_library_usage.py` file with comprehensive documentation.

## Important: Async Naming Convention

**All async functions now use the `_async` suffix.** This was implemented to maintain consistency and avoid naming conflicts. When using async functions, always include `_async` in the function name.

## Quick Start

```python
import asyncio
from mlBridgeLib.mlBridgeBPLib import (
    get_all_tournaments_async,
    get_tournament_clubs_dataframe_async,
    get_teams_by_tournament_async,
    get_board_results_by_team_async,
    get_boards_by_deal_async,
    get_all_boards_for_team_async
)

# Basic usage example
async def extract_tournament_data():
    tr = "S202602"  # Tournament ID
    cl = "5802079"  # Club ID
    sc = "A"        # Section
    eq = "212"      # Team number
    
    # Get all boards for a team
    boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    print(f"Extracted {len(boards_df)} boards")
    return boards_df, frequency_df

# Run the extraction
boards, frequency = asyncio.run(extract_tournament_data())
```

## Function Categories

### 1. Tournament Discovery Functions

#### `get_all_tournaments_async()`
Discover all available tournaments from the main results page.

```python
async def discover_tournaments():
    tournaments_df = await get_all_tournaments_async()
    print(f"Found {len(tournaments_df)} tournaments")
    
    # Show sample tournaments
    for i in range(min(3, len(tournaments_df))):
        row = tournaments_df.row(i, named=True)
        print(f"{row['Date']} - {row['Tournament_Name']} (ID: {row['Tournament_ID']})")
    
    return tournaments_df
```

**Returns:** DataFrame with columns: `Date`, `Tournament_ID`, `Tournament_Name`

#### `get_tournament_clubs_dataframe_async(tr)`
Get all clubs participating in a specific tournament.

```python
async def get_tournament_clubs():
    tr = "S202602"  # Tournament ID
    
    clubs_df = await get_tournament_clubs_dataframe_async(tr)
    print(f"Found {len(clubs_df)} clubs")
    
    # Show club information
    for i in range(min(3, len(clubs_df))):
        row = clubs_df.row(i, named=True)
        print(f"Club {row['Club_ID']}: {row['Club_Name']}")
    
    return clubs_df
```

**Returns:** DataFrame with columns: `Date`, `Tournament_ID`, `Event_Name`, `Club_ID`, `Club_Name`

#### `get_teams_by_tournament_async(tr, cl)`
Get team rankings for a specific tournament and club.

```python
async def get_teams():
    tr = "S202602"  # Tournament ID
    cl = "5802079"  # Club ID
    
    teams_df = await get_teams_by_tournament_async(tr, cl)
    print(f"Found {len(teams_df)} teams")
    
    # Show top teams
    for i in range(min(3, len(teams_df))):
        row = teams_df.row(i, named=True)
        print(f"Rank {row['Rank']}: {row['Player1_Name']} - {row['Player2_Name']}")
        print(f"  Score: {row['Percent']}%, Team {row['Team_Number']}, Section {row['Section']}")
    
    return teams_df
```

**Returns:** DataFrame with columns: `Rank`, `Percent`, `Points`, `Bonus`, `Player1_ID`, `Player2_ID`, `Player1_Name`, `Player2_Name`, `Section`, `Team_Number`, `Event_Name`, `Date`, `Team_Count`, `Club_Count`, `Event_ID`, `Club_ID`

### 2. Board Results Functions

#### `get_board_results_by_team_async(tr, cl, sc, eq)`
Get board results for a specific team (their route through the tournament).

```python
async def get_team_results():
    tr = "S202602"  # Tournament ID
    cl = "5802079"  # Club ID
    sc = "A"        # Section
    eq = "212"      # Team number
    
    board_results_df = await get_board_results_by_team_async(tr, cl, sc, eq)
    print(f"Found {len(board_results_df)} board results")
    
    # Show sample results
    for i in range(min(3, len(board_results_df))):
        row = board_results_df.row(i, named=True)
        print(f"Board {row['Board']}: {row['Contract']} by {row['Declarer']}")
        print(f"  Score: {row['Score']}, Percentage: {row['Percentage']}%")
        print(f"  Opponent: {row['Opponent_Pair_Names']} ({row['Opponent_Pair_Direction']}{row['Opponent_Pair_Number']} {row['Opponent_Pair_Section']})")
    
    return board_results_df
```

**Returns:** DataFrame with columns: `Board`, `Score`, `Percentage`, `Contract`, `Declarer`, `Lead`, `Pair_Names`, `Pair_Direction`, `Pair_Number`, `Opponent_Pair_Names`, `Opponent_Pair_Direction`, `Opponent_Pair_Number`, `Opponent_Pair_Section`

### 3. Board Detail Functions

#### `get_boards_by_deal_async(tr, cl, sc, eq, d)`
Get detailed information for a single board/deal.

```python
async def get_single_board():
    tr = "S202602"  # Tournament ID
    cl = "5802079"  # Club ID
    sc = "A"        # Section
    eq = "212"      # Team number
    d = "2"         # Deal number
    
    boards_result = await get_boards_by_deal_async(tr, cl, sc, eq, d)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    print(f"Board data: {len(boards_df)} records")
    print(f"Frequency data: {len(frequency_df)} records")
    
    # Show board details
    if len(boards_df) > 0:
        row = boards_df.row(0, named=True)
        print(f"Board {row['Board']}: {row['Contract']} by {row['Declarer']}")
        print(f"Result: {row['Result']}, Score: {row['Score']}")
        print(f"PBN: {row['PBN']}")
        print(f"Opponent: {row['Opponent_Pair_Names']} ({row['Opponent_Pair_Direction']}{row['Opponent_Pair_Number']} {row['Opponent_Pair_Section']})")
    
    return boards_result
```

**Returns:** Dictionary with:
- `boards`: DataFrame with columns: `Board`, `PBN`, `Contract`, `Result`, `Lead`, `Dealer`, `Vul`, `Team_Name`, `Opponent_Pair_Names`, `Opponent_Pair_Direction`, `Opponent_Pair_Number`, `Opponent_Pair_Section`, `Score`
- `score_frequency`: DataFrame with columns: `Board`, `Score`, `Frequency`, `Matchpoints_NS`, `Matchpoints_EW`

#### `get_all_boards_for_team_async(tr, cl, sc, eq)` ⭐ Most Used
Get all boards played by a specific team (recommended for most use cases).

```python
async def get_all_team_boards():
    tr = "S202602"  # Tournament ID
    cl = "5802079"  # Club ID
    sc = "A"        # Section
    eq = "212"      # Team number
    
    boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    print(f"Extracted {len(boards_df)} boards")
    print(f"Frequency data: {len(frequency_df)} records")
    
    # Validate PBN quality
    valid_pbn_count = sum(1 for i in range(len(boards_df)) 
                         if boards_df.row(i, named=True)['PBN'].startswith('N:'))
    print(f"PBN quality: {valid_pbn_count}/{len(boards_df)} valid ({valid_pbn_count/len(boards_df)*100:.1f}%)")
    
    return boards_result
```

**Returns:** Same format as `get_boards_by_deal_async()` but for all boards played by the team.

### 4. Player-Specific Functions

#### `get_board_for_player_async(tr, cl, player_id, d)`
Get detailed information for a single board from a player's perspective.

```python
async def get_player_board():
    tr = "S202602"      # Tournament ID
    cl = "5802079"      # Club ID
    player_id = "12345" # Player ID
    d = "2"             # Deal number
    
    boards_result = await get_board_for_player_async(tr, cl, player_id, d)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    if len(boards_df) > 0:
        row = boards_df.row(0, named=True)
        print(f"Player board {row['Board']}: {row['Contract']} by {row['Declarer']}")
        print(f"Result: {row['Result']}, Score: {row['Score']}")
        print(f"Opponent: {row['Opponent_Pair_Names']} ({row['Opponent_Pair_Direction']}{row['Opponent_Pair_Number']} {row['Opponent_Pair_Section']})")
    
    return boards_result
```

**Returns:** Same format as `get_boards_by_deal_async()` but from player perspective. Only includes boards actually played by the player (boards not played are skipped entirely).

#### `get_all_boards_for_player_async(tr, cl, player_id)`
Get all boards for a specific player (only boards that player actually played).

```python
async def get_all_player_boards():
    tr = "S202602"      # Tournament ID
    cl = "5802079"      # Club ID
    player_id = "12345" # Player ID
    
    boards_result = await get_all_boards_for_player_async(tr, cl, player_id)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    print(f"Player played {len(boards_df)} boards")
    print(f"Frequency data: {len(frequency_df)} records")
    
    # Show player's performance
    if len(boards_df) > 0:
        avg_score = sum(row['Score'] for row in boards_df.iter_rows(named=True)) / len(boards_df)
        print(f"Average score: {avg_score:.1f}")
    
    return boards_result
```

**Returns:** Same format as `get_boards_by_deal_async()` but for all boards played by the player.

### 5. Advanced Board Retrieval Functions

#### `get_all_boards_async(tr, cl, max_deals=36)`
Get ALL boards from 1 to max_deals, regardless of which teams played them.

```python
async def get_all_tournament_boards():
    tr = "S202602"  # Tournament ID
    cl = "5802079"  # Club ID
    
    boards_result = await get_all_boards_async(tr, cl, max_deals=36)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    print(f"Found {len(boards_df)} boards in tournament")
    return boards_result
```

**Behavior:** Uses first available team to try to get every board 1-36. Returns all boards that exist in the tournament.

#### `get_board_by_number_async(tr, cl, board_number)`
Get ONE specific board by number (searches through teams to find one that played it).

```python
async def get_specific_board():
    tr = "S202602"     # Tournament ID
    cl = "5802079"     # Club ID
    board_number = 5   # Board number to find
    
    boards_result = await get_board_by_number_async(tr, cl, board_number)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    print(f"Found board {board_number} data")
    return boards_result
```

**Behavior:** Searches through all teams until it finds one that played this board. Returns data from first team found.

### 6. Non-Async Wrapper Functions

All async functions have non-async wrappers for simpler usage:

```python
# Non-async versions
tournaments_df = get_all_tournaments()
clubs_df = get_tournament_clubs_dataframe(tr)
teams_df = get_teams_by_tournament(tr, cl)
board_results_df = get_board_results_by_team(tr, cl, sc, eq)
boards_result = get_boards_by_deal(tr, cl, sc, eq, d)
boards_result = get_all_boards_for_team(tr, cl, sc, eq)
boards_result = get_all_boards_for_player(tr, cl, player_id)
boards_result = get_board_for_player(tr, cl, player_id, d)
boards_result = get_board_for_team(tr, cl, sc, eq, d)
boards_result = get_all_boards(tr, cl, max_deals=36)
boards_result = get_board_by_number(tr, cl, board_number)
```

### 7. High-Performance Combined Functions

#### `request_complete_tournament_data_async(teams_url, board_results_url, boards_url)`
Extract teams, board results, and boards data simultaneously with shared browser context.

```python
async def get_complete_data():
    tr = "S202602"
    cl = "5802079"
    sc = "A"
    eq = "212"
    d = "2"
    
    # Build URLs
    teams_url = f"https://www.bridgeplus.com/nos-simultanes/resultats/?p=club&res=sim&tr={tr}&cl={cl}"
    board_results_url = f"https://www.bridgeplus.com/nos-simultanes/resultats/?p=route&res=sim&tr={tr}&cl={cl}&sc={sc}&eq={eq}"
    boards_url = f"https://www.bridgeplus.com/nos-simultanes/resultats/?p=donne&res=sim&d={d}&eq={eq}&tr={tr}&cl={cl}&sc={sc}"
    
    # Extract all data simultaneously
    result = await request_complete_tournament_data_async(teams_url, board_results_url, boards_url)
    
    teams_df = result['teams']
    board_results_df = result['board_results']
    boards_df = result['boards']
    frequency_df = result['score_frequency']
    
    print(f"Teams: {teams_df.shape}")
    print(f"Board Results: {board_results_df.shape}")
    print(f"Boards: {boards_df.shape}")
    print(f"Frequency: {frequency_df.shape}")
    
    return result
```

**Returns:** Dictionary with `teams`, `board_results`, `boards`, and `score_frequency` DataFrames.

## Complete Usage Examples

### Example 1: Tournament Analysis Pipeline

```python
import asyncio
from mlBridgeLib.mlBridgeBPLib import (
    get_all_tournaments_async,
    get_tournament_clubs_dataframe_async,
    get_teams_by_tournament_async,
    get_all_boards_for_team_async
)

async def analyze_tournament():
    # 1. Discover tournaments
    tournaments_df = await get_all_tournaments_async()
    print(f"Found {len(tournaments_df)} tournaments")
    
    # 2. Pick a tournament
    tr = tournaments_df['Tournament_ID'][0]  # Use first tournament
    
    # 3. Get clubs in tournament
    clubs_df = await get_tournament_clubs_dataframe_async(tr)
    print(f"Found {len(clubs_df)} clubs in tournament {tr}")
    
    # 4. Pick a club
    cl = clubs_df['Club_ID'][0]  # Use first club
    
    # 5. Get teams in club
    teams_df = await get_teams_by_tournament_async(tr, cl)
    print(f"Found {len(teams_df)} teams in club {cl}")
    
    # 6. Analyze top team
    top_team = teams_df.row(0, named=True)
    sc = top_team['Section']
    eq = top_team['Team_Number']
    
    # 7. Get all boards for top team
    boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
    boards_df = boards_result['boards']
    
    print(f"Top team: {top_team['Player1_Name']} - {top_team['Player2_Name']}")
    print(f"Percentage: {top_team['Percent']}%")
    print(f"Boards played: {len(boards_df)}")
    
    return {
        'tournament': tr,
        'club': cl,
        'team': top_team,
        'boards': boards_df
    }

# Run analysis
result = asyncio.run(analyze_tournament())
```

### Example 2: Player Performance Analysis

```python
async def analyze_player_performance():
    tr = "S202602"
    cl = "5802079"
    player_id = "12345"
    
    # Get all boards for player
    boards_result = await get_all_boards_for_player_async(tr, cl, player_id)
    boards_df = boards_result['boards']
    
    if len(boards_df) == 0:
        print(f"No boards found for player {player_id}")
        return
    
    # Calculate performance metrics
    total_boards = len(boards_df)
    avg_score = sum(row['Score'] for row in boards_df.iter_rows(named=True)) / total_boards
    
    # Contract analysis
    contracts = [row['Contract'] for row in boards_df.iter_rows(named=True)]
    game_contracts = [c for c in contracts if c.startswith(('3N', '4', '5', '6', '7'))]
    
    print(f"Player {player_id} Performance:")
    print(f"  Total boards: {total_boards}")
    print(f"  Average score: {avg_score:.1f}")
    print(f"  Game contracts: {len(game_contracts)}/{total_boards} ({len(game_contracts)/total_boards*100:.1f}%)")
    
    return {
        'player_id': player_id,
        'boards': boards_df,
        'avg_score': avg_score,
        'total_boards': total_boards
    }

# Run player analysis
player_result = asyncio.run(analyze_player_performance())
```

### Example 3: Data Export and Persistence

```python
async def export_tournament_data():
    tr = "S202602"
    cl = "5802079"
    sc = "A"
    eq = "212"
    
    # Get comprehensive data
    boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
    boards_df = boards_result['boards']
    frequency_df = boards_result['score_frequency']
    
    # Export to multiple formats
    
    # CSV for spreadsheet analysis
    boards_df.write_csv(f"tournament_{tr}_club_{cl}_boards.csv")
    frequency_df.write_csv(f"tournament_{tr}_club_{cl}_frequency.csv")
    
    # JSON for structured data
    boards_df.write_json(f"tournament_{tr}_club_{cl}_boards.json")
    frequency_df.write_json(f"tournament_{tr}_club_{cl}_frequency.json")
    
    # Parquet for efficient storage
    boards_df.write_parquet(f"tournament_{tr}_club_{cl}_boards.parquet")
    frequency_df.write_parquet(f"tournament_{tr}_club_{cl}_frequency.parquet")
    
    print(f"Data exported for tournament {tr}, club {cl}")
    print(f"Files created: CSV, JSON, and Parquet formats")
    
    return True

# Run export
asyncio.run(export_tournament_data())
```

## Best Practices

### 1. Error Handling

The library now includes enhanced error handling with detailed diagnostics for common issues:

```python
async def safe_extraction():
    try:
        boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
        boards_df = boards_result['boards']
        
        if len(boards_df) == 0:
            print("No boards found - team may not have played any boards")
            return None
        
        return boards_result
        
    except ValueError as e:
        if "not found in section" in str(e):
            print(f"Team validation error: {e}")
            # The error message includes available teams for debugging
        elif "No route data found" in str(e):
            print(f"Route data error: {e}")
            # Team exists but has no route data
        else:
            print(f"Validation error: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 2. Performance Optimization

```python
async def optimized_extraction():
    # Use shared browser context for multiple operations
    async with get_browser_context_async() as context:
        # All operations will reuse the same browser
        teams_task = get_teams_by_tournament_async(tr, cl)
        board_results_task = get_board_results_by_team_async(tr, cl, sc, eq)
        
        # Run in parallel
        teams_df, board_results_df = await asyncio.gather(teams_task, board_results_task)
    
    return teams_df, board_results_df
```

### 3. Data Validation

```python
async def validate_data():
    boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
    boards_df = boards_result['boards']
    
    # Validate PBN format
    valid_pbn_count = 0
    for row in boards_df.iter_rows(named=True):
        if row['PBN'].startswith('N:') and len(row['PBN']) > 50:
            valid_pbn_count += 1
    
    print(f"PBN validation: {valid_pbn_count}/{len(boards_df)} valid")
    
    # Validate opponent information
    complete_opponent_info = 0
    for row in boards_df.iter_rows(named=True):
        if (row['Opponent_Pair_Names'] != 'Unknown' and
            row['Opponent_Pair_Direction'] != 'Unknown' and
            row['Opponent_Pair_Number'] != 0 and
            row['Opponent_Pair_Section'] != 'Unknown'):
            complete_opponent_info += 1
    
    print(f"Opponent info: {complete_opponent_info}/{len(boards_df)} complete")
    
    return boards_result
```

### 4. Data Schema Improvements

#### New Opponent Information Fields

All DataFrames now include comprehensive opponent information:

```python
# Board Results and Boards DataFrames now include:
- Opponent_Pair_Direction: 'NS' or 'EW' 
- Opponent_Pair_Number: Team number (e.g., 113)
- Opponent_Pair_Names: Player names (e.g., 'SININGE - KRIEF')
- Opponent_Pair_Section: Section letter (e.g., 'A')
```

#### Example Data Access

```python
boards_result = await get_all_boards_for_team_async(tr, cl, sc, eq)
boards_df = boards_result['boards']

# Access opponent information
for row in boards_df.iter_rows(named=True):
    print(f"Board {row['Board']}: vs {row['Opponent_Pair_Names']}")
    print(f"  Direction: {row['Opponent_Pair_Direction']}")
    print(f"  Number: {row['Opponent_Pair_Number']}")
    print(f"  Section: {row['Opponent_Pair_Section']}")
```

#### Data Validation Features

The library now includes automatic validation:
- **Section Consistency**: Warns when opponent section doesn't match expected values
- **Data Integrity**: Validates opponent information extraction from HTML
- **Field Completeness**: Ensures all opponent fields are populated or marked as 'Unknown'
- **Team Existence**: Automatically checks if teams exist before attempting to extract data
- **Route Data Validation**: Provides detailed diagnostics when route data is missing

## Testing

Use the unified test suite to verify library functionality:

```bash
python test_bridge_library.py
```

This will test all functions and provide a comprehensive report of library health.

## Performance Notes

- **Parallel extraction**: 2-3x faster than individual requests
- **Shared browser context**: Reduces overhead for multiple requests
- **Intelligent waiting**: No fixed delays, adapts to website response times
- **French to English translation**: Automatic conversion of contracts, directions, and cards
- **PBN format**: Validated bridge deal format for compatibility with bridge software
- **Enhanced parsing**: Dynamic board number extraction from HTML instead of hardcoded assumptions
- **Robust opponent extraction**: Uses specific HTML div structure for reliable opponent information

## Technical Improvements

### 1. Enhanced HTML Parsing

The library now uses more robust HTML parsing strategies:

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

### 2. Improved Opponent Information Extraction

The library now uses the specific HTML structure for opponent information:

```python
# Uses specific div structure: <div class="col">contre <span class="paires">...
opponent_div = await page.query_selector('div.col:has-text("contre")')
if opponent_div:
    paires_span = await opponent_div.query_selector('span.paires')
    if paires_span:
        # Parse opponent text like "NS : SININGE - KRIEF (113 A)"
        paires_text = paires_span.text_content()
        # Extract direction, names, number, and section
```

### 3. Better Error Handling and Diagnostics

The library provides detailed diagnostic information:

```python
# Enhanced error messages
if len(route_data) < 5:
    logger.warning(f"Very few route records extracted: only {len(route_data)} records found")
    logger.warning("This might indicate:")
    logger.warning("  1. Team played few boards")
    logger.warning("  2. HTML parsing issues")
    logger.warning("  3. Page structure changed")
```

### 4. Intelligent Data Extraction

The library now:
- Extracts board numbers dynamically from HTML content
- Maintains correspondence between board numbers, scores, and matchpoints
- Validates data integrity and provides warnings for unusual patterns
- Handles edge cases like void suits in PBN format
- Provides graduated warnings instead of hard failures

## Migration Notes

### From Previous Versions

If you were using the library before the latest improvements:

- **Enhanced opponent information**: All DataFrames now include `Opponent_Pair_Section` field
- **Better error handling**: More graceful handling of missing data with detailed diagnostics
- **Dynamic board parsing**: No longer assumes fixed board numbers (19, 20 skip pattern removed)
- **Improved HTML parsing**: More robust extraction using actual page structure
- **Section validation**: Automatic detection of opponent section mismatches

### Function Compatibility

All existing function signatures remain the same. The improvements are internal enhancements that provide:
- More reliable data extraction
- Better error messages
- Enhanced data validation
- Improved performance

## Changelog

### Version 2.2.0 (Latest)
