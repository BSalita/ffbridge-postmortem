import asyncio
import re
import os
import sys
import json
import requests
from pathlib import Path
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from dotenv import load_dotenv, set_key
from datetime import datetime

async def fetch_easi_token_with_lancelot(lancelot_token):
    """
    Fetch EASI token using Lancelot bearer token
    Similar to what api-easi-token.bat does
    """
    try:
        print("Fetching EASI token using Lancelot token...")
        
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9,fr;q=0.8',
            'authorization': f'Bearer {lancelot_token}',
            'dnt': '1',
            'origin': 'https://www.ffbridge.fr',
            'referer': 'https://www.ffbridge.fr/',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
        }
        
        response = requests.get(
            'https://api-lancelot.ffbridge.fr/users/me/easi-token',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            # Response might be JSON or plain text
            try:
                data = response.json()
                if isinstance(data, dict):
                    # Look for token in common fields
                    for field in ['token', 'easi_token', 'easiToken', 'access_token']:
                        if field in data:
                            print(f"Found EASI token in field '{field}'")
                            return data[field]
                    # If no obvious field, return the whole response
                    print("EASI token found but in unexpected format")
                    return str(data)
                else:
                    print("EASI token response is not a dict")
                    return str(data)
            except json.JSONDecodeError:
                # Response is plain text (probably the token itself)
                token = response.text.strip()
                if len(token) > 20:  # Reasonable token length
                    print(f"Found EASI token as plain text: {token[:20]}...")
                    return token
                else:
                    print(f"EASI token response too short: {token}")
                    return None
        else:
            print(f"Failed to fetch EASI token: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error fetching EASI token: {e}")
        return None

async def extract_bearer_token_from_network(page):
    """Extract Bearer token from network requests"""
    bearer_tokens = []
    
    async def handle_request(request):
        headers = await request.all_headers()
        if 'authorization' in headers:
            auth_header = headers['authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.replace('Bearer ', '')
                bearer_tokens.append({
                    'token': token,
                    'url': request.url,
                    'method': request.method
                })
                print(f"Found Bearer token in request to: {request.url}")
    
    async def handle_response(response):
        headers = await response.all_headers()
        # Check for auth tokens in response headers
        for header_name, header_value in headers.items():
            if 'token' in header_name.lower() or 'authorization' in header_name.lower():
                print(f"Found auth-related response header: {header_name}: {header_value}")
        
        # Check response body for tokens (for JSON responses)
        if 'application/json' in headers.get('content-type', ''):
            try:
                body = await response.text()
                body_json = json.loads(body)
                
                # Look for common token fields
                token_fields = ['token', 'access_token', 'bearer_token', 'auth_token', 'authorization']
                for field in token_fields:
                    if field in body_json:
                        print(f"Found token in response body: {field}: {body_json[field]}")
                        bearer_tokens.append({
                            'token': body_json[field],
                            'url': response.url,
                            'method': 'RESPONSE_BODY',
                            'field': field
                        })
            except (json.JSONDecodeError, Exception):
                pass
    
    page.on('request', handle_request)
    page.on('response', handle_response)
    
    return bearer_tokens

def get_env_path():
    """Get path to .env file in current directory"""
    return Path.cwd() / ".env"

def load_or_create_env():
    """Load .env file or create it if missing, return config dict"""
    env_path = get_env_path()
    
    # Load existing .env file
    load_dotenv(env_path)
    
    config = {
        'email': os.getenv('FFBRIDGE_EMAIL'),
        'password': os.getenv('FFBRIDGE_PASSWORD'),
        'headless': os.getenv('FFBRIDGE_HEADLESS', 'true').lower() == 'true',
        'bearer_token': os.getenv('FFBRIDGE_BEARER_TOKEN_LANCELOT')
    }
    
    # Check if we need to prompt for missing values
    missing_values = []
    if not config['email']:
        missing_values.append('email')
    if not config['password']:
        missing_values.append('password')
    
    if missing_values:
        print(f"\n📝 .env file setup required")
        print(f"Missing values: {', '.join(missing_values)}")
        print(f"Location: {env_path}")
        
        if not env_path.exists():
            print("Creating new .env file...")
            env_path.touch()
        
        # Prompt for missing values
        if not config['email']:
            config['email'] = input("\nEnter your FFBridge email: ").strip()
            set_key(env_path, 'FFBRIDGE_EMAIL', config['email'])
        
        if not config['password']:
            import getpass
            config['password'] = getpass.getpass("Enter your FFBridge password: ").strip()
            set_key(env_path, 'FFBRIDGE_PASSWORD', config['password'])
        
        # Set default headless if not already set
        if not os.getenv('FFBRIDGE_HEADLESS'):
            headless_choice = input("Run in headless mode by default? (y/n): ").lower() == 'y'
            set_key(env_path, 'FFBRIDGE_HEADLESS', str(headless_choice).lower())
            config['headless'] = headless_choice
        
        print(f"✅ Configuration saved to {env_path}")
    
    return config

def save_bearer_tokens(bearer_tokens):
    """Save Bearer tokens to .env file"""
    env_path = get_env_path()
    
    # Separate tokens by type and domain
    lancelot_tokens = [t for t in bearer_tokens if 'api-lancelot.ffbridge.fr' in t.get('domain', '') or t.get('type') == 'lancelot']
    easi_tokens = [t for t in bearer_tokens if t.get('type') == 'easi' or 'easi-token' in t.get('url', '')]
    
    tokens_saved = 0
    
    if lancelot_tokens:
        # Use the latest lancelot token
        latest_lancelot = max(lancelot_tokens, key=lambda x: x.get('timestamp', 0))
        set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LANCELOT', latest_lancelot['token'])
        print(f"Lancelot Bearer token saved: {latest_lancelot['token'][:20]}...")
        tokens_saved += 1
    
    if easi_tokens:
        # Use the latest EASI token
        latest_easi = max(easi_tokens, key=lambda x: x.get('timestamp', 0))
        set_key(env_path, 'FFBRIDGE_EASI_TOKEN', latest_easi['token'])
        print(f"EASI token saved: {latest_easi['token'][:20]}...")
        tokens_saved += 1
    
    # Always set timestamp when tokens are processed
    if bearer_tokens:
        timestamp = datetime.now().isoformat()
        set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LAST_UPDATE', timestamp)
        print(f"Token update timestamp set: {timestamp}")
    
    if tokens_saved > 0:
        print(f"{tokens_saved} Bearer token(s) saved to {env_path}")
        return True
    else:
        print(f"No tokens saved to {env_path}")
        return False

def save_bearer_token(token):
    """Legacy function - save single bearer token to .env file"""
    env_path = get_env_path()
    set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LANCELOT', token)
    print(f"Bearer token saved to {env_path}")

async def get_ffbridge_bearer_token_playwright(username=None, password=None, headless=None):
    """Get FFBridge Bearer token using Playwright browser automation"""
    
    config = load_or_create_env()
    
    # Use provided values or fall back to config
    username = username or config['email']
    password = password or config['password']
    headless = headless if headless is not None else config['headless']
    
    if not username or not password:
        print("❌ Username and password are required")
        return None
    
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Track network requests to capture Bearer tokens
        bearer_tokens = []
        
        async def handle_request(request):
            headers = request.headers
            if 'authorization' in headers:
                auth_header = headers['authorization']
                if auth_header.startswith('Bearer '):
                    token = auth_header.replace('Bearer ', '')
                    bearer_tokens.append({
                        'token': token,
                        'url': request.url,
                        'timestamp': datetime.now().timestamp(),
                        'domain': request.url.split('/')[2] if '/' in request.url else 'unknown'
                    })
                    print(f"Found Bearer token in request to: {request.url}")
        
        async def handle_response(response):
            headers = response.headers
            for header_name, header_value in headers.items():
                if 'token' in header_name.lower() or 'authorization' in header_name.lower():
                    print(f"Found auth header in response: {header_name}: {header_value}")
            
            # Also check response body for tokens
            try:
                if 'json' in response.headers.get('content-type', ''):
                    response_data = await response.json()
                    if isinstance(response_data, dict):
                        # Look for common token field names
                        token_fields = ['token', 'access_token', 'bearer_token', 'auth_token', 'authorization', 'easiToken', 'easi_token']
                        for field in token_fields:
                            if field in response_data:
                                # Determine token type based on URL and field name
                                token_type = 'general'
                                if 'easi-token' in response.url or 'easi' in field.lower():
                                    token_type = 'easi'
                                elif 'api-lancelot' in response.url:
                                    token_type = 'lancelot'
                                
                                bearer_tokens.append({
                                    'token': response_data[field],
                                    'url': response.url,
                                    'source': f'response_body.{field}',
                                    'timestamp': datetime.now().timestamp(),
                                    'domain': response.url.split('/')[2] if '/' in response.url else 'unknown',
                                    'type': token_type
                                })
                                print(f"Found {token_type} token in response body: {field} from {response.url}")
                        
                        # Special case: if this is the easi-token endpoint, check for any token-like values
                        if 'easi-token' in response.url:
                            # The response might be just the token string directly
                            if isinstance(response_data, str) and len(response_data) > 20:
                                bearer_tokens.append({
                                    'token': response_data,
                                    'url': response.url,
                                    'source': 'response_body.direct',
                                    'timestamp': datetime.now().timestamp(),
                                    'domain': response.url.split('/')[2] if '/' in response.url else 'unknown',
                                    'type': 'easi'
                                })
                                print(f"Found EASI token as direct response from {response.url}")
                            # Or check for any field that looks like a token
                            for key, value in response_data.items():
                                if isinstance(value, str) and len(value) > 20 and ('token' in key.lower() or len(value) > 100):
                                    bearer_tokens.append({
                                        'token': value,
                                        'url': response.url,
                                        'source': f'response_body.{key}',
                                        'timestamp': datetime.now().timestamp(),
                                        'domain': response.url.split('/')[2] if '/' in response.url else 'unknown',
                                        'type': 'easi'
                                    })
                                    print(f"Found EASI token in field {key} from {response.url}")
            except:
                pass  # Ignore JSON parsing errors
        
        # Enable request/response interception
        page.on('request', handle_request)
        page.on('response', handle_response)
        
        try:
            print("Navigating to ffbridge login page...")
            await page.goto('https://licencie.ffbridge.fr', wait_until='networkidle')
            
            # Take screenshot for debugging
            await page.screenshot(path='login_page_playwright.png')
            
            print("Waiting for login form...")
            await page.wait_for_timeout(2000)
            
            # Look for username/email field with multiple strategies
            username_selectors = [
                'input[type="email"]',
                'input[name="email"]', 
                'input[name="username"]',
                'input[placeholder*="email" i]',
                'input[placeholder*="utilisateur" i]',
                '#email',
                '#username'
            ]
            
            username_field = None
            for selector in username_selectors:
                try:
                    username_field = await page.wait_for_selector(selector, timeout=2000)
                    if username_field:
                        print(f"Found username field with selector: {selector}")
                        break
                except:
                    continue
            
            if not username_field:
                print("Could not find username field")
                await page.screenshot(path='no_username_field.png')
                return None
            
            # Look for password field
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[placeholder*="mot de passe" i]',
                'input[placeholder*="password" i]',
                '#password'
            ]
            
            password_field = None
            for selector in password_selectors:
                try:
                    password_field = await page.wait_for_selector(selector, timeout=2000)
                    if password_field:
                        print(f"Found password field with selector: {selector}")
                        break
                except:
                    continue
            
            if not password_field:
                print("Could not find password field")
                await page.screenshot(path='no_password_field.png')
                return None
            
            print("Filling credentials...")
            await username_field.fill(username)
            await password_field.fill(password)
            
            # Look for login button
            login_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Connexion")',
                'button:has-text("Se connecter")',
                'button:has-text("Login")',
                '.btn-primary',
                '#login-button'
            ]
            
            login_button = None
            for selector in login_selectors:
                try:
                    login_button = await page.wait_for_selector(selector, timeout=2000)
                    if login_button:
                        print(f"Found login button with selector: {selector}")
                        break
                except:
                    continue
            
            if not login_button:
                print("Could not find login button")
                await page.screenshot(path='no_login_button.png')
                return None
            
            print("Clicking login button...")
            await login_button.click()
            
            # Wait for login to complete and page to load
            print("Waiting for login to complete...")
            await page.wait_for_timeout(3000)
            
            # CRITICAL: Click the "Accéder" button to access "Mon espace licencié" 
            # This should generate additional token requests
            print("Looking for 'Mon espace licencié' access button...")
            try:
                # Look for the "Accéder" button in the "Mon espace licencié" section
                access_button_selectors = [
                    'button:has-text("Accéder")',
                    'button[class*="tui-island__footer-button"]:has-text("Accéder")',
                    'button[data-appearance="flat"]:has-text("Accéder")',
                    'button:has-text("Mon espace licencié")',
                    'a:has-text("Accéder")',
                    '[class*="tui-island"] button',
                ]
                
                access_button = None
                for selector in access_button_selectors:
                    try:
                        access_button = await page.wait_for_selector(selector, timeout=5000)
                        if access_button:
                            # Check if this is the right button by looking for nearby text
                            button_area = await page.query_selector(f'{selector}:near(:text("Mon espace licencié"))')
                            if button_area or "accéder" in (await access_button.text_content()).lower():
                                print(f"Found 'Accéder' button with selector: {selector}")
                                break
                    except:
                        continue
                
                if access_button:
                    print("Clicking 'Accéder' button to access Mon espace licencié...")
                    await access_button.click()
                    
                    # Wait for navigation to the licensed area
                    print("Waiting for navigation to licensed area...")
                    await page.wait_for_timeout(5000)
                    
                    # This should trigger additional API calls
                    print("Waiting for additional token generation...")
                    await page.wait_for_timeout(3000)
                    
                else:
                    print("Could not find 'Accéder' button, trying to navigate directly...")
                    # Try to navigate directly to the licensed area
                    await page.goto('https://licencie.ffbridge.fr', wait_until='networkidle')
                    await page.wait_for_timeout(3000)
                    
            except Exception as e:
                print(f"Error accessing licensed area: {e}")
                # Fallback - just wait and see if tokens are generated anyway
                await page.wait_for_timeout(5000)
            
            # Now try to trigger more API calls
            print("Trying to trigger additional API calls...")
            try:
                # Look for navigation elements that might trigger additional API calls
                nav_links = await page.query_selector_all('a[href*="resultats"], a[href*="tournoi"], a[href*="simultane"], button, .nav-link, .menu-item')
                for link in nav_links[:3]:  # Try first 3 links
                    try:
                        text = await link.text_content()
                        if text and any(word in text.lower() for word in ['résultat', 'tournoi', 'simultané', 'classement']):
                            print(f"Clicking on navigation: {text[:30]}...")
                            await link.click()
                            await page.wait_for_timeout(3000)
                            break
                    except:
                        continue
            except Exception as e:
                print(f"Error navigating: {e}")
            
            print("Login completed, checking for tokens...")
            await page.wait_for_timeout(2000)
            
            if bearer_tokens:
                print(f"\nFound {len(bearer_tokens)} Bearer tokens:")
                for i, token_info in enumerate(bearer_tokens):
                    domain = token_info.get('domain', 'unknown')
                    url = token_info.get('url', 'unknown')
                    print(f"  {i+1}. Domain: {domain}")
                    print(f"     URL: {url}")
                    print(f"     Token: {token_info['token'][:20]}...")
                
                # Save all tokens to .env file with domain-specific keys
                save_bearer_tokens(bearer_tokens)
                
                # ACTIVELY FETCH EASI TOKEN using Lancelot token
                lancelot_tokens = [t for t in bearer_tokens if 'api-lancelot.ffbridge.fr' in t.get('domain', '')]
                if lancelot_tokens:
                    latest_lancelot = max(lancelot_tokens, key=lambda x: x.get('timestamp', 0))
                    easi_token = await fetch_easi_token_with_lancelot(latest_lancelot['token'])
                    
                    if easi_token:
                        # Add the EASI token to our bearer_tokens list
                        bearer_tokens.append({
                            'token': easi_token,
                            'url': 'https://api-lancelot.ffbridge.fr/users/me/easi-token',
                            'source': 'active_fetch',
                            'timestamp': datetime.now().timestamp(),
                            'domain': 'api-lancelot.ffbridge.fr',
                            'type': 'easi'
                        })
                        
                        # Save the EASI token to .env
                        env_path = get_env_path()
                        set_key(env_path, 'FFBRIDGE_EASI_TOKEN', easi_token)
                        print(f"EASI token saved: {easi_token[:20]}...")
                        
                        # Update timestamp
                        timestamp = datetime.now().isoformat()
                        set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LAST_UPDATE', timestamp)
                
                # PRIORITIZE tokens from api-lancelot.ffbridge.fr domain
                lancelot_tokens = [t for t in bearer_tokens if 'api-lancelot.ffbridge.fr' in t.get('domain', '')]
                
                if lancelot_tokens:
                    print(f"\nFound {len(lancelot_tokens)} tokens from api-lancelot.ffbridge.fr domain!")
                    latest_token = max(lancelot_tokens, key=lambda x: x.get('timestamp', 0))
                    token = latest_token['token']
                    print(f"Using api-lancelot.ffbridge.fr token: {token[:20]}...")
                    return token
                else:
                    print(f"\nNo domain-specific tokens found.")
                    # Show all domains found
                    domains = set(t.get('domain', 'unknown') for t in bearer_tokens)
                    print(f"Domains found: {list(domains)}")
                    
                    # Use the latest token anyway for testing
                    latest_token = max(bearer_tokens, key=lambda x: x.get('timestamp', 0))
                    token = latest_token['token']
                    domain = latest_token.get('domain', 'unknown')
                    print(f"Using latest token from {domain}: {token[:20]}...")
                    return token
            else:
                print("No Bearer tokens found in network traffic")
                await page.screenshot(path='no_tokens_found.png')
                return None
                
        except Exception as e:
            print(f"Error during login: {e}")
            await page.screenshot(path='login_error.png')
            return None
        finally:
            await browser.close()

def get_bearer_token_playwright_sync(username=None, password=None, headless=None):
    """Synchronous wrapper for the async function"""
    return asyncio.run(get_ffbridge_bearer_token_playwright(username, password, headless))

async def get_bearer_token_interactive_playwright():
    """Interactive Playwright version using .env configuration"""
    print("FFBridge Bearer Token Extractor (Playwright)")
    print("=" * 45)
    
    # Load or create .env configuration
    config = load_or_create_env()
    
    # Check if we already have a valid token
    if config['bearer_token']:
        print(f"\nFound existing Bearer token in .env file")
        print(f"Token (first 20 chars): {config['bearer_token'][:20]}...")
        
        use_existing = input("Use existing token? (y/n): ").lower() == 'y'
        if use_existing:
            print(f"\nUsing existing token from .env file")
            print(f"Bearer Token: {config['bearer_token']}")
            return config['bearer_token']
    
    print(f"\nStarting browser automation...")
    print(f"Email: {config['email']}")
    print(f"Headless mode: {config['headless']}")
    
    token = await get_ffbridge_bearer_token_playwright()
    
    if token:
        print(f"\nSUCCESS!")
        print(f"Bearer Token: {token}")
        print(f"\nYou can now use this token in your API calls:")
        print(f'headers = {{"Authorization": "Bearer {token}"}}')
        return token
    else:
        print(f"\nFAILED to extract Bearer token")
        return None

def get_bearer_token_from_env(quiet=False):
    """Simply get the bearer token from .env file without automation"""
    load_dotenv()
    token = os.getenv('FFBRIDGE_BEARER_TOKEN_LANCELOT')
    if token:
        if not quiet:
            print("Bearer token loaded from .env file")
        return token
    else:
        if not quiet:
            print("No bearer token found in .env file")
            print("Run the automation script first to obtain a token")
        return None

def update_env_with_token(token, domain='lancelot'):
    """
    Update .env file with a bearer token for a specific domain
    
    Args:
        token: Bearer token to save
        domain: Domain type ('lancelot', 'easi') - default: 'lancelot'
    
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> import ffbridge_auth_playwright as ffauth
        >>> success = ffauth.update_env_with_token('eyJ...', 'lancelot')
        >>> print(f"Updated: {success}")
    """
    try:
        env_path = get_env_path()
        
        if domain == 'lancelot':
            set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LANCELOT', token)
        elif domain == 'easi':
            set_key(env_path, 'FFBRIDGE_EASI_TOKEN', token)
        else:
            # Default to lancelot token
            set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LANCELOT', token)
        
        # Update timestamp
        timestamp = datetime.now().isoformat()
        set_key(env_path, 'FFBRIDGE_BEARER_TOKEN_LAST_UPDATE', timestamp)
        
        print(f"Token updated in .env file for domain: {domain}")
        return True
        
    except Exception as e:
        print(f"Error updating .env file: {e}")
        return False

def main():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FFBridge Bearer Token Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Automatic mode (default) - get new tokens and update .env
  %(prog)s --interactive            # Interactive mode with prompts
  %(prog)s --from-env               # Get token from .env file only
  %(prog)s --update-env --token "eyJ..." # Update .env file with provided token
  %(prog)s --email user@domain.com  # Specify email (will prompt for password)
  %(prog)s --headless false         # Run with visible browser
  %(prog)s --quiet                  # Minimal output, just print token
        """
    )
    
    parser.add_argument('--mode', choices=['interactive', 'auto', 'from-env', 'update-env'], 
                       default='auto',
                       help='Operation mode (default: auto)')
    parser.add_argument('--from-env', action='store_true',
                       help='Just read token from .env file (no automation)')
    parser.add_argument('--auto', action='store_true', 
                       help='Automatic mode using .env credentials')
    parser.add_argument('--update-env', action='store_true',
                       help='Update .env file with provided token')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with prompts')
    parser.add_argument('--email', type=str,
                       help='FFBridge email address')
    parser.add_argument('--password', type=str,
                       help='FFBridge password (not recommended, use .env file)')
    parser.add_argument('--token', type=str,
                       help='Bearer token to save to .env file (for update-env mode)')
    parser.add_argument('--headless', type=str, choices=['true', 'false'], 
                       help='Run browser in headless mode (true/false)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - only output the token')
    parser.add_argument('--output', '-o', type=str,
                       help='Save token to specified file')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.from_env:
        mode = 'from-env'
    elif args.auto:
        mode = 'auto'
    elif args.update_env:
        mode = 'update-env'
    elif args.interactive:
        mode = 'interactive'
    else:
        mode = args.mode
    
    # Set headless preference
    headless = None
    if args.headless:
        headless = args.headless.lower() == 'true'
    
    try:
        if mode == 'from-env':
            # Just read from .env file
            if not args.quiet:
                print("Reading token from .env file...")
            token = get_bearer_token_from_env(quiet=args.quiet)
            
        elif mode == 'auto':
            # Automatic mode using .env credentials
            if not args.quiet:
                print("Automatic mode - using .env credentials...")
            token = get_bearer_token_playwright_sync(
                username=args.email, 
                password=args.password, 
                headless=headless
            )
            
        elif mode == 'update-env':
            # Update .env file mode
            if not args.token:
                print("Error: --token is required for update-env mode", file=sys.stderr)
                return 1
            
            if not args.quiet:
                print("Updating .env file with provided token...")
            
            # Save the token to .env file
            save_bearer_token(args.token)
            
            if not args.quiet:
                print(f"Token saved to .env file: {args.token[:20]}...")
            
            token = args.token
            
        else:  # interactive
            # Interactive mode
            if args.quiet:
                print("Warning: Interactive mode cannot be quiet", file=sys.stderr)
            token = asyncio.run(get_bearer_token_interactive_playwright())
        
        # Output result
        if token:
            if args.quiet:
                print(token)
            else:
                print(f"\nSuccess! Bearer Token: {token}")
                
            # Save to file if requested
            if args.output:
                try:
                    with open(args.output, 'w') as f:
                        f.write(token)
                    if not args.quiet:
                        print(f"Token saved to {args.output}")
                except Exception as e:
                    print(f"Failed to save token to {args.output}: {e}", file=sys.stderr)
                    return 1
            
            return 0
        else:
            if not args.quiet:
                print("Failed to get Bearer token")
            return 1
            
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
        return 1

# Public API functions for import use
def get_token_sync(username=None, password=None, headless=True):
    """
    Synchronous function to get FFBridge token - for import use
    
    Args:
        username: FFBridge email (optional, will use .env if not provided)
        password: FFBridge password (optional, will use .env if not provided) 
        headless: Run browser in headless mode (default: True)
        
    Returns:
        str: Bearer token if successful, None if failed
        
    Example:
        >>> import ffbridge_auth_playwright as ffauth
        >>> token = ffauth.get_token_sync('user@domain.com', 'password')
        >>> print(f"Token: {token}")
    """
    return get_bearer_token_playwright_sync(username, password, headless)

async def get_token_async(username=None, password=None, headless=True):
    """
    Async function to get FFBridge token - for import use
    
    Args:
        username: FFBridge email (optional, will use .env if not provided)
        password: FFBridge password (optional, will use .env if not provided)
        headless: Run browser in headless mode (default: True)
        
    Returns:
        str: Bearer token if successful, None if failed
        
    Example:
        >>> import ffbridge_auth_playwright as ffauth
        >>> token = await ffauth.get_token_async('user@domain.com', 'password')
        >>> print(f"Token: {token}")
    """
    return await get_ffbridge_bearer_token_playwright(username, password, headless)

def get_token_from_file(env_file='.env'):
    """
    Get token from .env file without automation - for import use
    
    Args:
        env_file: Path to .env file (default: '.env')
        
    Returns:
        str: Bearer token if found, None if not found
        
    Example:
        >>> import ffbridge_auth_playwright as ffauth
        >>> token = ffauth.get_token_from_file()
        >>> print(f"Token: {token}")
    """
    if env_file != '.env':
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    return os.getenv('FFBRIDGE_BEARER_TOKEN_LANCELOT')

if __name__ == "__main__":
    import sys
    sys.exit(main()) 