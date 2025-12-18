import logging
import os
import json
from http.server import BaseHTTPRequestHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from google import genai
from google.genai import types
from google.genai.errors import APIError
from google.api_core.exceptions import ServiceUnavailable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio

# Load from environment variables (secure)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"

# Pro users
PRO_USERS = {6094195032, 7463224424, 7214569577}

# Global Gemini client
gemini_client = None
RETRY_EXCEPTIONS = (ServiceUnavailable,)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# WARNING: In-memory state resets on Vercel cold starts
# For production, use Redis/Database
user_states = {}

# --- Gemini Initialization ---

def init_gemini_client():
    """Initializes the Gemini client using the configured API key."""
    global gemini_client
    try:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set!")
            return False
        
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini Client Initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")
        return False

# --- Gemini API Calls ---

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    reraise=True
)
def _attempt_gemini_call(prompt: str, system_prompt: str, max_tokens: int):
    global gemini_client
    if not gemini_client:
        raise Exception("Gemini client is not available.")

    temperature = 0.7 if max_tokens < 2000 else 0.8

    if system_prompt:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature
        )
    else:
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
    
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=config
    )
    
    if not response.text:
        if response.candidates and response.candidates[0].safety_ratings:
            raise APIError("Gemini generation failed due to safety filters.")
        raise APIError("Gemini returned empty content.")

    return response.text

def call_gemini(prompt: str, system_prompt: str = "", max_tokens: int = 4000):
    if not gemini_client:
        return "Initialization Error: Gemini client is not available."
    
    try:
        return _attempt_gemini_call(prompt, system_prompt, max_tokens)
    except ServiceUnavailable:
        logger.error("Gemini API Error: 503 UNAVAILABLE.")
        return "Gemini API Error: The model is temporarily overloaded. Please try again."
    except APIError as e:
        logger.error(f"Gemini API Error: {e}")
        return f"Gemini API Error: {e}"
    except Exception as e:
        logger.error(f"Internal Error calling Gemini: {e}")
        return f"Internal Error: {e}"

# --- Helper Functions ---

def is_pro_user(user_id):
    return user_id in PRO_USERS

def clean_output(text):
    text = text.replace('**', '').replace('__', '').replace('*', '')
    text = text.replace('===', '').replace('---', '').replace('```', '')
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text.strip()

async def send_pro_upgrade_message(update: Update, context: ContextTypes.DEFAULT_TYPE, feature_name: str):
    keyboard = [
        [InlineKeyboardButton("🔓 Unlock PRO Features", url="https://t.me/your_pro_channel")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    target = update.callback_query if update.callback_query else update.message
    message_text = (
        f"🚫 {feature_name} is a PRO Feature 💎\n\n"
        "Upgrade to PRO to access:\n"
        "• AI Content Consultant\n"
        "• Scout Bounties\n"
        "• Full-length premium content\n\n"
        "Ready to go viral?"
    )

    if update.callback_query:
        await target.edit_message_text(message_text, reply_markup=reply_markup)
    else:
        await target.reply_text(message_text, reply_markup=reply_markup)

# --- Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    user_id = update.message.from_user.id
    is_pro = is_pro_user(user_id)
    
    if is_pro:
        status = "PRO USER 💎"
        keyboard = [
            [InlineKeyboardButton("✨ Analyze Content", callback_data='analyze')],
            [InlineKeyboardButton("🎨 Generate Content", callback_data='generate')],
            [InlineKeyboardButton("💡 AI Consultant 💎", callback_data='chat')],
            [InlineKeyboardButton("🎯 Scout Bounties 💎", callback_data='bounties')],
            [InlineKeyboardButton("❓ Help", callback_data='help')]
        ]
    else:
        status = "Free Tier 🆓 (Limited Output)"
        keyboard = [
            [InlineKeyboardButton("✨ Analyze Content (Draft)", callback_data='analyze')],
            [InlineKeyboardButton("🎨 Generate Content (Draft)", callback_data='generate')],
            [InlineKeyboardButton("🔥 Upgrade to PRO 💎", url="https://t.me/your_pro_channel")],
            [InlineKeyboardButton("❓ Help", callback_data='help')]
        ]
        
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🚀 Welcome to GoViral Pro\n"
        f"Status: {status} (ID: {user_id})\n\n"
        f"Your AI-powered content creator and analyzer.\n\n"
        f"What would you like to do today?",
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    await update.message.reply_text(
        "❓ How to Use GoViral Pro\n\n"
        "COMMANDS:\n"
        "/start - Main menu\n"
        "/analyze - Analyze your content (get scores + 5 hook variations)\n"
        "/generate - Create full articles or threads\n"
        "/chat - AI Content Consultant (PRO) - Full conversation mode\n"
        "/bounties - Scout viral contests (PRO)\n"
        "/skip - Get topic ideas (during Generate mode)\n"
        "/cancel - Cancel current action\n\n"
        "HOW IT WORKS:\n\n"
        "ANALYZE:\n"
        "1. Paste your content\n"
        "2. Get viral scores and 5 hook alternatives\n"
        "3. Choose to rewrite it or not\n\n"
        "GENERATE:\n"
        "1. Pick format (article/thread)\n"
        "2. Choose niche, tone, mood\n"
        "3. Enter your topic OR use /skip for ideas\n"
        "4. Get complete ready-to-post content\n\n"
        "CONSULTANT (PRO):\n"
        "Full conversation mode - ask anything about content strategy, growth, monetization. It remembers context!"
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Analyze mode"""
    user_id = update.message.from_user.id
    is_pro = is_pro_user(user_id)
    
    keyboard = [
        [InlineKeyboardButton("📄 Article", callback_data='analyze_article')],
        [InlineKeyboardButton("🧵 Thread", callback_data='analyze_thread')],
        [InlineKeyboardButton("← Back", callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    tier_label = "" if is_pro else " (Draft Level)"
    await update.message.reply_text(f"✨ Analyze Mode{tier_label}\n\nChoose format:", reply_markup=reply_markup)

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Generate mode"""
    user_id = update.message.from_user.id
    is_pro = is_pro_user(user_id)
        
    keyboard = [
        [InlineKeyboardButton("📄 Article", callback_data='gen_article')],
        [InlineKeyboardButton("🧵 Thread", callback_data='gen_thread')],
        [InlineKeyboardButton("← Back", callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    tier_label = "" if is_pro else " (Draft Level)"
    await update.message.reply_text(f"🎨 Generate Mode{tier_label}\n\nWhat format?", reply_markup=reply_markup)

async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Chat mode (PRO Only)"""
    user_id = update.message.from_user.id
    if not is_pro_user(user_id):
        await send_pro_upgrade_message(update, context, "AI Content Consultant")
        return
    
    user_states[user_id] = {'mode': 'chat', 'history': []}
    
    await update.message.reply_text(
        "💡 AI Content Consultant - PRO MODE 💎\n\n"
        "I am your dedicated AI Content Consultant.\n"
        "Ask me for strategy, critique, ideas, or analysis.\n\n"
        "Use /start to return to menu."
    )

async def bounties_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Bounties (PRO Only)"""
    user_id = update.message.from_user.id
    if not is_pro_user(user_id):
        await send_pro_upgrade_message(update, context, "Scout Bounties")
        return
        
    await update.message.reply_text(
        "🎯 Scout Bounties - PRO Feature 💎\n\n"
        "This feature helps you find and track X/Twitter thread contests and bounties.\n\n"
        "PRO FEATURES:\n"
        "• Real-time bounty alerts\n"
        "• Auto-scout trending contests\n"
        "• Prize tracking"
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel operation"""
    user_id = update.message.from_user.id
    
    if user_id in user_states:
        del user_states[user_id]
    await update.message.reply_text("❌ Cancelled. Use /start to begin again.")

async def skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Skip topic and get topic suggestions"""
    user_id = update.message.from_user.id
    is_pro = is_pro_user(user_id)
        
    if user_id in user_states and user_states[user_id]['mode'] == 'generate':
        format_type = user_states[user_id].get('format', 'article')
        niche = user_states[user_id].get('niche', 'entrepreneurship')
        tone = user_states[user_id].get('tone', 'authoritative')
        mood = user_states[user_id].get('mood', 'inspiring')
        
        await update.message.reply_text("⏳ Generating topic ideas... (10 seconds)")
        
        max_tokens = 1500 if is_pro else 800
        
        system_prompt = f"You are a content topic expert for {niche}. Generate trending, viral-worthy topic ideas. Be concise and actionable. No formatting symbols."
        
        prompt = f"""Generate 8 trending topic ideas for {format_type} in the {niche} niche.

Tone: {tone}
Mood: {mood}

For each topic (numbered 1-8):
- One compelling headline/hook
- One sentence on why it would perform well

Keep it tight and scannable."""

        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        await update.message.reply_text(cleaned)
        
        await update.message.reply_text(
            "💡 Pick a topic number (1-8) or describe your own topic, and I'll write the full content for you!"
        )
    else:
        await update.message.reply_text("The /skip command only works while in Generate mode.")

# --- Button Handler ---

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    is_pro = is_pro_user(user_id)
    data = query.data
    
    if data == 'analyze':
        tier_label = "" if is_pro else " (Draft Level)"
        keyboard = [
            [InlineKeyboardButton("📄 Article", callback_data='analyze_article')],
            [InlineKeyboardButton("🧵 Thread", callback_data='analyze_thread')],
            [InlineKeyboardButton("← Back", callback_data='back')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"✨ Analyze Mode{tier_label}\n\nChoose format:", reply_markup=reply_markup)
    
    elif data == 'generate':
        tier_label = "" if is_pro else " (Draft Level)"
        keyboard = [
            [InlineKeyboardButton("📄 Article", callback_data='gen_article')],
            [InlineKeyboardButton("🧵 Thread", callback_data='gen_thread')],
            [InlineKeyboardButton("← Back", callback_data='back')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"🎨 Generate Mode{tier_label}\n\nWhat format?", reply_markup=reply_markup)

    elif data == 'chat':
        if not is_pro:
            await send_pro_upgrade_message(update, context, "AI Content Consultant")
            return
        user_states[user_id] = {'mode': 'chat', 'history': []}
        await query.edit_message_text(
            "💡 AI Content Consultant - PRO MODE 💎\n\n"
            "I am your dedicated AI Content Consultant.\n"
            "Ask me for strategy, critique, ideas, or analysis.\n\n"
            "Use /start to return to menu."
        )
        
    elif data == 'bounties':
        if not is_pro:
            await send_pro_upgrade_message(update, context, "Scout Bounties")
            return
            
        await query.edit_message_text(
            "🎯 Scout Bounties - PRO Feature 💎\n\n"
            "This feature helps you find and track X/Twitter thread contests and bounties.\n\n"
            "PRO FEATURES:\n"
            "• Real-time bounty alerts\n"
            "• Auto-scout trending contests\n"
            "• Prize tracking",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data='back')]])
        )
        
    elif data == 'help':
        await query.message.reply_text(
            "❓ How to Use GoViral Pro\n\n"
            "COMMANDS:\n"
            "/start - Main menu\n"
            "/analyze - Analyze your content (get scores + 5 hook variations)\n"
            "/generate - Create full articles or threads\n"
            "/chat - AI Content Consultant (PRO) - Full conversation mode\n"
            "/bounties - Scout viral contests (PRO)\n"
            "/skip - Get topic ideas (during Generate mode)\n"
            "/cancel - Cancel current action\n\n"
            "HOW IT WORKS:\n\n"
            "ANALYZE:\n"
            "1. Paste your content\n"
            "2. Get viral scores and 5 hook alternatives\n"
            "3. Choose to rewrite it or not\n\n"
            "GENERATE:\n"
            "1. Pick format (article/thread)\n"
            "2. Choose niche, tone, mood\n"
            "3. Enter your topic OR use /skip for ideas\n"
            "4. Get complete ready-to-post content\n\n"
            "CONSULTANT (PRO):\n"
            "Full conversation mode - ask anything about content strategy, growth, monetization. It remembers context!"
        )
        
    elif data == 'back':
        is_pro = is_pro_user(user_id)
        
        if is_pro:
            status = "PRO USER 💎"
            keyboard = [
                [InlineKeyboardButton("✨ Analyze Content", callback_data='analyze')],
                [InlineKeyboardButton("🎨 Generate Content", callback_data='generate')],
                [InlineKeyboardButton("💡 AI Consultant 💎", callback_data='chat')],
                [InlineKeyboardButton("🎯 Scout Bounties 💎", callback_data='bounties')],
                [InlineKeyboardButton("❓ Help", callback_data='help')]
            ]
        else:
            status = "Free Tier 🆓 (Limited Output)"
            keyboard = [
                [InlineKeyboardButton("✨ Analyze Content (Draft)", callback_data='analyze')],
                [InlineKeyboardButton("🎨 Generate Content (Draft)", callback_data='generate')],
                [InlineKeyboardButton("🔥 Upgrade to PRO 💎", url="https://t.me/your_pro_channel")],
                [InlineKeyboardButton("❓ Help", callback_data='help')]
            ]
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"🚀 Welcome to GoViral Pro\n"
            f"Status: {status} (ID: {user_id})\n\n"
            f"Your AI-powered content creator and analyzer.\n\n"
            f"What would you like to do today?",
            reply_markup=reply_markup
        )

    elif data.startswith('analyze_'):
        format_type = data.split('_')[1]
        user_states[user_id] = {'mode': 'analyze', 'format': format_type}
        
        tier_note = "\n\n(Note: Free tier results are limited to 1500 tokens for brevity.)" if not is_pro else ""
        
        await query.edit_message_text(
            f"📄 {format_type.capitalize()} Analysis\n\n"
            "Paste your content now and I'll analyze:\n"
            "• Hook effectiveness\n"
            "• Emotional triggers\n"
            "• Viral potential\n"
            "• 10 alternative hooks"
            f"{tier_note}"
        )
        
    elif data == 'gen_improved':
        state = user_states.get(user_id)
        if not state or 'original_text' not in state:
            await query.edit_message_text("Error: Original content not found. Please restart analysis.")
            return

        format_type = state.get('format', 'article')
        original_text = state['original_text']
        
        await update.message.reply_text("⏳ Rewriting your content for maximum impact... (30 seconds)")
        
        max_tokens = 4000 if is_pro else 2000
        
        system_prompt = f"You are an elite content rewriter. Transform the user's {format_type} into a viral masterpiece. Fix weak hooks, improve flow, add emotional resonance, and maximize shareability. Write in clean, powerful prose without formatting symbols."
        
        prompt = f"""Rewrite this {format_type} to maximize its viral potential:

ORIGINAL CONTENT:
{original_text}

YOUR TASK:
Completely rewrite this to be MORE:
- Attention-grabbing (stronger hook)
- Valuable (clearer insights)
- Emotional (triggers curiosity, inspiration, or urgency)
- Shareable (quotable lines, memorable phrases)
- Actionable (clear takeaways)

Keep the core message but make it 10x more impactful. Write the complete rewritten {format_type} now."""
        
        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        if not is_pro:
            cleaned += "\n\n--- \n💡 This is a Free Tier draft. Upgrade to PRO for full, professional-grade content rewriting and optimization."

        chunks = [cleaned[i:i+3800] for i in range(0, len(cleaned), 3800)]
        for chunk in chunks:
            await query.message.reply_text(chunk)
            
        del user_states[user_id]
        keyboard = [[InlineKeyboardButton("🔄 Start New Analysis", callback_data='analyze')], [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("✅ Improved content generated! What next?", reply_markup=reply_markup)

    elif data == 'skip_improved':
        if user_id in user_states:
            del user_states[user_id]
        await query.edit_message_text("Understood. Analysis complete. Use /start to access the main menu.")

    elif data.startswith('gen_'):
        format_type = data.split('_')[1]
        user_states[user_id] = {'mode': 'generate', 'format': format_type}
        
        keyboard = [
            [InlineKeyboardButton("Entrepreneurship", callback_data=f'niche_entrepreneurship')],
            [InlineKeyboardButton("Tech", callback_data=f'niche_tech')],
            [InlineKeyboardButton("Finance", callback_data=f'niche_finance')],
            [InlineKeyboardButton("Growth", callback_data=f'niche_growth')],
            [InlineKeyboardButton("Social", callback_data=f'niche_social')],
            [InlineKeyboardButton("Creative", callback_data=f'niche_creative')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"🎨 Generate {format_type.capitalize()}\n\nSelect Niche:", reply_markup=reply_markup)
        
    elif data.startswith('niche_'):
        niche = data.split('_')[1]
        if user_id in user_states:
            user_states[user_id]['niche'] = niche
        
        keyboard = [
            [InlineKeyboardButton("Authoritative", callback_data=f'tone_authoritative')],
            [InlineKeyboardButton("Bold", callback_data=f'tone_bold')],
            [InlineKeyboardButton("Conversational", callback_data=f'tone_conversational')],
            [InlineKeyboardButton("Inspiring", callback_data=f'tone_inspiring')],
            [InlineKeyboardButton("Casual", callback_data=f'tone_casual')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"Niche set to {niche.capitalize()}.\n\nSelect Tone:", reply_markup=reply_markup)
        
    elif data.startswith('tone_'):
        tone = data.split('_')[1]
        if user_id in user_states:
            user_states[user_id]['tone'] = tone
            
        keyboard = [
            [InlineKeyboardButton("Curious", callback_data=f'mood_curious')],
            [InlineKeyboardButton("Urgent", callback_data=f'mood_urgent')],
            [InlineKeyboardButton("Reflective", callback_data=f'mood_reflective')],
            [InlineKeyboardButton("Educational", callback_data=f'mood_educational')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"Tone set to {tone.capitalize()}.\n\nSelect Mood:", reply_markup=reply_markup)
        
    elif data.startswith('mood_'):
        mood = data.split('_')[1]
        state = user_states.get(user_id)
        if user_id in user_states:
            user_states[user_id]['mood'] = mood
            
            format_type = state.get('format', 'content')
            niche = state.get('niche', 'general')
            tone = state.get('tone', 'neutral')
            
            tier_note = "\n\n(Note: Free tier results are limited to 1500 tokens. Upgrade to PRO for full-length, professional output.)" if not is_pro else ""

            await query.edit_message_text(
                f"✅ Ready to Generate {format_type.capitalize()}\n"
                f"Niche: {niche.capitalize()}, Tone: {tone.capitalize()}, Mood: {mood.capitalize()}.\n\n"
                f"Now, send me the topic or hook you want content written about.\n"
                f"Use /skip to generate content ideas instead."
                f"{tier_note}"
            )

# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages"""
    user_id = update.message.from_user.id
    text = update.message.text
    is_pro = is_pro_user(user_id)
    
    # Chat mode (AI Content Consultant - PRO Only)
    if user_id in user_states and user_states[user_id].get('mode') == 'chat':
        if not is_pro:
            await send_pro_upgrade_message(update, context, "AI Content Consultant")
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Advanced consultant with conversation memory and deep expertise
        history = user_states[user_id].get('history', [])
        history.append(f"User: {text}")
        
        # Build context from conversation history
        context_str = "\n".join(history[-10:])  # Last 10 exchanges
        
        system_prompt = """You are an elite AI Content Strategist and Consultant - the absolute best in the industry.

YOUR EXPERTISE:
- Viral content psychology and mechanics
- Platform algorithms (X/Twitter, Instagram, TikTok, YouTube)
- Audience growth and engagement strategies
- Monetization tactics for creators
- Hook writing and storytelling frameworks
- Content calendars and posting strategies
- Analytics interpretation and optimization
- Personal branding and positioning

YOUR APPROACH:
- Be conversational but highly knowledgeable
- Give specific, actionable advice (not generic tips)
- Use examples and case studies when helpful
- Ask clarifying questions if needed to give better advice
- Reference current trends and what's working NOW
- Be honest about what won't work
- Provide frameworks and step-by-step guidance
- Anticipate follow-up questions

YOUR STYLE:
- Direct and confident (you're an expert)
- Friendly and encouraging
- No fluff or filler
- Practical over theoretical
- Use bullet points for clarity when listing multiple items
- Keep responses focused but comprehensive (150-300 words unless asked for more detail)

REMEMBER:
- You have access to the conversation history
- Build on previous exchanges
- Remember user's goals and context
- Proactively suggest next steps

Never discuss technical details about the bot itself, code, or internal systems. Stay focused on content strategy only."""

        full_prompt = f"""Conversation history:
{context_str}

Respond to the user's latest message with expert strategic guidance."""
        
        response = call_gemini(full_prompt, system_prompt, max_tokens=2000)
        cleaned = clean_output(response)
        
        # Add response to history
        history.append(f"Consultant: {cleaned}")
        user_states[user_id]['history'] = history
        
        keyboard = [
            [InlineKeyboardButton("✍️ Generate Content", callback_data='generate')],
            [InlineKeyboardButton("🔍 Analyze Content", callback_data='analyze')],
            [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        suggestion = "\n\n---\nReady to put this into action? Use the buttons below."
        
        final_message = cleaned + suggestion
        
        max_length = 3800
        chunks = [final_message[i:i+max_length] for i in range(0, len(final_message), max_length)]
        
        for i in range(len(chunks) - 1):
            await update.message.reply_text(chunks[i])

        if chunks:
            await update.message.reply_text(chunks[-1], reply_markup=reply_markup)
            
        return
    
    # Smart detection for random messages when no state is set
    if user_id not in user_states:
        if any(word in text.lower() for word in ['?', 'how', 'what', 'why', 'can you', 'please', 'help', 'tell me', 'show me', 'explain']):
            await update.message.reply_text(
                "👋 Hi! I detected a question.\n\n"
                "Use /start to access the main menu and use my features."
            )
        else:
            await update.message.reply_text("Use /start to begin! 🚀")
        return
    
    state = user_states[user_id]
    
    max_tokens = 3500 if is_pro else 1500
    
    # Analyze mode (Free/PRO)
    if state['mode'] == 'analyze':
        format_type = state.get('format', 'article')
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        max_tokens = 2500 if is_pro else 1500
        
        if format_type == 'article':
            system_prompt = "You are an expert content analyst. Analyze viral potential with precision. Use percentage scores. Write in clean, flowing prose without formatting symbols."
            prompt = f"""Analyze this article for viral potential:

{text}

Provide analysis in this order:

1. HOOK EFFECTIVENESS SCORE: X%
Brief explanation of why

2. EMOTIONAL IMPACT SCORE: X%
Which emotions are triggered and how strong

3. VIRAL POTENTIAL SCORE: X%
Overall assessment

4. TOP 3 STRENGTHS
What works well

5. TOP 3 IMPROVEMENTS NEEDED
What could be better

6. 5 ALTERNATIVE HOOKS
Give me 5 different hook variations that could work better, numbered 1-5.

Be direct and actionable."""
            
        else:  # thread
            system_prompt = "You are an expert at analyzing viral Twitter/X threads. Provide concise, actionable analysis with percentage scores. No formatting symbols."
            prompt = f"""Analyze this thread for viral potential:

{text}

Provide analysis in this order:

1. HOOK EFFECTIVENESS SCORE: X%
Why it works or doesn't

2. THREAD FLOW SCORE: X%
How well tweets connect

3. VIRAL POTENTIAL SCORE: X%
Overall assessment

4. TOP 3 STRENGTHS
What works well

5. TOP 3 IMPROVEMENTS NEEDED
What could be better

6. 5 ALTERNATIVE OPENING HOOKS
Better ways to start this thread, numbered 1-5.

Be concise and actionable."""
            
        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        if not is_pro:
            cleaned += "\n\n---\n💡 Free Tier: Limited analysis depth. Upgrade to PRO for comprehensive critique and strategic insights."
            
        chunks = [cleaned[i:i+3800] for i in range(0, len(cleaned), 3800)]
        
        for i, chunk in enumerate(chunks):
            if i > 0: 
                await update.message.reply_text(f"...continued ({i+1}/{len(chunks)})")
            await update.message.reply_text(chunk)
        
        user_states[user_id] = {'mode': 'ask_improved', 'format': format_type, 'original_text': text}
        keyboard = [
            [InlineKeyboardButton("✅ Yes, Rewrite It Better", callback_data='gen_improved')], 
            [InlineKeyboardButton("❌ No Thanks", callback_data='skip_improved')], 
            [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Would you like me to rewrite this content to maximize its viral potential?", 
            reply_markup=reply_markup
        )
    
    # Generate mode (Free/PRO)
    elif state['mode'] == 'generate':
        format_type = state.get('format', 'article')
        niche = state.get('niche', 'entrepreneurship')
        tone = state.get('tone', 'authoritative')
        mood = state.get('mood', 'inspiring')
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        max_tokens = 4000 if is_pro else 2000
        
        if format_type == 'article':
            system_prompt = f"You are an elite content creator specializing in viral {niche} articles. Write compelling, well-structured articles with strong hooks, clear value, and emotional resonance. Write in {tone} tone with {mood} mood. No formatting symbols - pure flowing prose."
            
            prompt = f"""Write a complete, publication-ready article on this topic:

TOPIC: {text}

REQUIREMENTS:
- Niche: {niche}
- Tone: {tone}
- Mood: {mood}
- Target: Viral social media performance

STRUCTURE:
1. Powerful opening hook (2-3 sentences that grab attention immediately)
2. Promise/value proposition (why keep reading)
3. 4-6 main sections with concrete insights, examples, or stories
4. Actionable takeaways
5. Strong closing with call-to-action

Write the FULL article now. Make it shareable, quotable, and valuable."""
            
        else:  # thread
            system_prompt = f"You are a viral Twitter/X thread expert for {niche}. Create threads that stop the scroll, deliver value, and get shared widely. Write in {tone} tone with {mood} mood. No formatting symbols."
            
            prompt = f"""Write a complete Twitter/X thread on this topic:

TOPIC: {text}

REQUIREMENTS:
- Niche: {niche}
- Tone: {tone}
- Mood: {mood}
- Target: Maximum engagement and virality

STRUCTURE:
Tweet 1: Attention-grabbing hook (must stop the scroll)
Tweets 2-7: Value-packed content (insights, stories, data, examples)
Tweet 8: Strong CTA (like, retweet, follow, comment)

Format each tweet like this:
1/8: [content]
2/8: [content]
etc.

Write the COMPLETE thread now (7-10 tweets). Make every tweet count."""
            
        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        if not is_pro:
            cleaned += "\n\n---\n💡 Free Tier: Shortened version. Upgrade to PRO for full-length, deeply detailed content with richer examples and insights."
            
        chunks = [cleaned[i:i+3800] for i in range(0, len(cleaned), 3800)]
        for chunk in chunks:
            await update.message.reply_text(chunk)
        
        del user_states[user_id]
        keyboard = [
            [InlineKeyboardButton("🔄 Generate Another", callback_data='generate')], 
            [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("✅ Content created! What's next?", reply_markup=reply_markup)

# Initialize application
application = None

def get_application():
    """Initialize application if not already done"""
    global application
    if application is None:
        if not TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_TOKEN environment variable not set")
        
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("analyze", analyze_command))
        application.add_handler(CommandHandler("generate", generate_command))
        application.add_handler(CommandHandler("chat", chat_command))
        application.add_handler(CommandHandler("bounties", bounties_command))
        application.add_handler(CommandHandler("cancel", cancel))
        application.add_handler(CommandHandler("skip", skip))
        application.add_handler(CallbackQueryHandler(button_callback))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Initialize Gemini
        init_gemini_client()
        
        logger.info("Application initialized successfully")
    
    return application

# Vercel serverless handler
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle incoming webhook POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        
        if content_length == 0:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad Request: Empty body')
            return
        
        try:
            post_data = self.rfile.read(content_length)
            update_data = json.loads(post_data.decode('utf-8'))
            
            # Get application instance
            app = get_application()
            
            # Create update object
            update = Update.de_json(update_data, app.bot)
            
            # Process update asynchronously
            async def process():
                async with app:
                    await app.process_update(update)
            
            # Run the async function
            asyncio.run(process())
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad Request: Invalid JSON')
            
        except Exception as e:
            logger.error(f"Error processing update: {e}", exc_info=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'Internal Server Error: {str(e)}'.encode())
    
    def do_GET(self):
        """Handle GET requests (health check)"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        status_html = """
        <!DOCTYPE html>
        <html>
        <head><title>GoViral Bot Status</title></head>
        <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1>🚀 GoViral Bot is Running!</h1>
            <p>Webhook endpoint is active and ready to receive updates.</p>
            <p style="color: #666; font-size: 14px;">This bot is deployed on Vercel</p>
        </body>
        </html>
        """
        self.wfile.write(status_html.encode())
