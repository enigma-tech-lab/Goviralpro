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
        f"🚫 **{feature_name} is a PRO Feature** 💎\n\n"
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
        "❓ **How to Use GoViral Pro**\n\n"
        "**Shortcuts:**\n"
        "/analyze - Start analyzing content (Free)\n"
        "/generate - Start generating content (Free)\n"
        "/chat - Talk to the AI Consultant (PRO)\n"
        "/bounties - View Bounty Scout (PRO)\n"
        "/cancel - Cancel current action\n"
        "/skip - Generate ideas (in Generate Mode)\n\n"
        "**Features:**\n"
        "✨ Analyze: Improve hooks & virality (Free/PRO)\n"
        "🎨 Generate: Create articles & threads (Free/PRO)\n"
        "💡 Consultant: Strategic advice (PRO Only)\n"
        "🎯 Bounties: Find contests (PRO Only)"
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
    await update.message.reply_text(f"✨ **Analyze Mode**{tier_label}\n\nChoose format:", reply_markup=reply_markup)

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
    await update.message.reply_text(f"🎨 **Generate Mode**{tier_label}\n\nWhat format?", reply_markup=reply_markup)

async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Chat mode (PRO Only)"""
    user_id = update.message.from_user.id
    if not is_pro_user(user_id):
        await send_pro_upgrade_message(update, context, "AI Content Consultant")
        return
    
    user_states[user_id] = {'mode': 'chat', 'history': []}
    
    await update.message.reply_text(
        "💡 **AI Content Consultant - PRO MODE 💎**\n\n"
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
        "🎯 **Scout Bounties - PRO Feature 💎**\n\n"
        "This feature helps you find and track X/Twitter thread contests and bounties.\n\n"
        "**PRO features:**\n"
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
    """Skip topic and generate ideas"""
    user_id = update.message.from_user.id
    is_pro = is_pro_user(user_id)
        
    if user_id in user_states and user_states[user_id]['mode'] == 'generate':
        format_type = user_states[user_id].get('format', 'article')
        niche = user_states[user_id].get('niche', 'entrepreneurship')
        
        await update.message.reply_text("⏳ Generating fresh ideas... (15-20 seconds)")
        
        max_tokens = 3000 if is_pro else 1500
        
        system_prompt = f"You are a content ideation expert. Your sole purpose is to generate ideas for content and nothing else. Generate compelling viral {format_type} ideas for {niche} niche. Write cleanly without formatting symbols."
        
        prompt = f"""Generate 10 viral {format_type} ideas for {niche} niche.

For each idea number them 1-10 and provide:

Hook or headline

Brief outline in 2-3 sentences

Why it would go viral

Make output clean and well organized."""

        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        if not is_pro:
            cleaned += "\n\n--- \n💡 This is a Free Tier draft. Upgrade to PRO for full-length, highly detailed ideation (up to 3x longer responses) and the AI Consultant."
            
        max_length = 3800
        chunks = [cleaned[i:i+max_length] for i in range(0, len(cleaned), max_length)]
        
        for chunk in chunks:
            await update.message.reply_text(chunk)
        
        del user_states[user_id]
        
        keyboard = [[InlineKeyboardButton("🔄 Generate More Ideas", callback_data='generate')],
                   [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("✅ Ideas generated! What next?", reply_markup=reply_markup)
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
            "💡 **AI Content Consultant - PRO MODE 💎**\n\n"
            "I am your dedicated AI Content Consultant.\n"
            "Ask me for strategy, critique, ideas, or analysis.\n\n"
            "Use /start to return to menu."
        )
        
    elif data == 'bounties':
        if not is_pro:
            await send_pro_upgrade_message(update, context, "Scout Bounties")
            return
            
        await query.edit_message_text(
            "🎯 **Scout Bounties - PRO Feature 💎**\n\n"
            "This feature helps you find and track X/Twitter thread contests and bounties.\n\n"
            "**PRO features:**\n"
            "• Real-time bounty alerts\n"
            "• Auto-scout trending contests\n"
            "• Prize tracking",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data='back')]])
        )
        
    elif data == 'help':
        await query.message.reply_text(
            "❓ **How to Use GoViral Pro**\n\n"
            "**Shortcuts:**\n"
            "/analyze - Start analyzing content (Free)\n"
            "/generate - Start generating content (Free)\n"
            "/chat - Talk to the AI Consultant (PRO)\n"
            "/bounties - View Bounty Scout (PRO)\n"
            "/cancel - Cancel current action\n"
            "/skip - Generate ideas (in Generate Mode)\n\n"
            "**Features:**\n"
            "✨ Analyze: Improve hooks & virality (Free/PRO)\n"
            "🎨 Generate: Create articles & threads (Free/PRO)\n"
            "💡 Consultant: Strategic advice (PRO Only)\n"
            "🎯 Bounties: Find contests (PRO Only)"
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
        
        await query.message.reply_text("⏳ Generating IMPROVED high-quality content... (30-45 seconds)")
        
        max_tokens = 3500 if is_pro else 1500
        
        system_prompt = f"You are an expert content optimizer. Your sole function is rewriting content. Rewrite the user's {format_type} to maximize its viral potential, addressing common errors and strengthening the hook. Do not use formatting symbols. Write as flowing prose."
        
        prompt = f"""Rewrite the following {format_type} for maximum virality and impact:\n\n{original_text}"""
        
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
        await query.edit_message_text(f"Niche set to **{niche.capitalize()}**.\n\nSelect Tone:", reply_markup=reply_markup, parse_mode='Markdown')
        
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
        await query.edit_message_text(f"Tone set to **{tone.capitalize()}**.\n\nSelect Mood:", reply_markup=reply_markup, parse_mode='Markdown')
        
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
                f"✅ Ready to Generate **{format_type.capitalize()}**\n"
                f"Niche: {niche.capitalize()}, Tone: {tone.capitalize()}, Mood: {mood.capitalize()}.\n\n"
                f"Now, send me the **topic** or **hook** you want content written about.\n"
                f"Use /skip to generate content ideas instead."
                f"{tier_note}"
            , parse_mode='Markdown')

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
        
        system_prompt = (
            "You are an expert, high-impact, strategic AI Content Consultant for the 'GoViral Bot'. "
            "Your function is strictly advisory on CONTENT and STRATEGY. "
            "Keep ALL responses under 150 words unless the user explicitly asks for a detailed analysis. "
            "NEVER discuss or reveal technical details, code structure, API calls, internal configuration, or user IDs. "
            "If asked about technical details, politely decline, stating: 'As the Content Consultant, my focus is purely on strategy, not the bot's backend structure. How can I help you with your content strategy?'"
            "Use bullet points for readability. Be conversational."
        )
        
        response = call_gemini(text, system_prompt, max_tokens=1000)
        cleaned = clean_output(response)
        
        keyboard = [
            [InlineKeyboardButton("1. Generate Content ✍️", callback_data='generate')],
            [InlineKeyboardButton("2. Analyze Content 🔍", callback_data='analyze')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        proactive_suggestion = (
            "\n\n---"
            "\nReady to bring this advice to life? "
            "Choose a tool below to take the next step."
        )
        
        final_message = cleaned + proactive_suggestion
        
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
        
        if format_type == 'article':
            system_prompt = "You are an expert content analyst specializing in viral content. Your only task is analysis. Analyze concisely and provide percentage-based ratings. Write in clean prose without any formatting symbols. Do not discuss your role or identity."
            prompt = f"""Analyze this article for virality potential: {text}

Provide:
1. Hook Effectiveness (0-100%)
2. Emotional Triggers (0-100%)
3. Viral Potential Score (0-100%)
4. Top 3 Strengths
5. Top 3 Weaknesses
6. 10 Alternative Hooks (numbered 1-10)

Be concise and actionable."""
            
        else:  # thread
            system_prompt = "You are an expert at analyzing viral Twitter/X threads. Your only task is analysis. Provide concise percentage-based analysis. Write cleanly without formatting symbols. Do not discuss your role or identity."
            prompt = f"""Analyze this thread for viral potential: {text}

Provide:
1. Hook Effectiveness (0-100%)
2. Thread Flow Score (0-100%)
3. Viral Potential (0-100%)
4. Top 3 Strengths
5. Top 3 Weaknesses
6. 10 Alternative Opening Hooks (numbered 1-10)

Be concise."""
            
        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        if not is_pro:
            cleaned += "\n\n--- \n💡 This is a Free Tier analysis (limited length). Upgrade to PRO for deep-dive, professional critique."
            
        chunks = [cleaned[i:i+3800] for i in range(0, len(cleaned), 3800)]
        
        for i, chunk in enumerate(chunks):
            if i > 0: await update.message.reply_text(f"...continued ({i+1}/{len(chunks)})")
            await update.message.reply_text(chunk)
        
        user_states[user_id] = {'mode': 'ask_improved', 'format': format_type, 'original_text': text}
        keyboard = [[InlineKeyboardButton("✅ Yes, Generate Improved Version", callback_data='gen_improved')], [InlineKeyboardButton("❌ No, I'm Good", callback_data='skip_improved')], [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("📝 Would you like me to generate an improved version?", reply_markup=reply_markup)
    
    # Generate mode (Free/PRO)
    elif state['mode'] == 'generate':
        format_type = state.get('format', 'article')
        niche = state.get('niche', 'entrepreneurship')
        tone = state.get('tone', 'authoritative')
        mood = state.get('mood', 'inspiring')
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        if format_type == 'article':
            system_prompt = f"You are an expert content creator specializing in viral {niche} content. Your sole purpose is to create the requested content. Write in a {tone} tone with a {mood} mood. Create premium, professional content. Never use formatting symbols. Write as flowing prose."
            prompt = f"""Create a viral article on: {text}

Target: {niche} audience
Tone: {tone}
Mood: {mood}

Structure:
- Compelling hook (first 2-3 sentences)
- 5-7 main points with supporting details
- Emotional triggers throughout
- Strong call-to-action ending

Make it shareable and engaging."""
            
        else:  # thread
            system_prompt = f"You are an expert at creating viral Twitter/X threads in {niche}. Your sole purpose is to create the requested thread. Write with {tone} tone and {mood} mood. Create engaging shareable content. Never use formatting symbols."
            prompt = f"""Create a viral Twitter/X thread on: {text}

Target: {niche} audience
Tone: {tone}
Mood: {mood}

Create ONE complete 5-10 tweet thread:
- Tweet 1: Hook (must grab attention)
- Tweets 2-8: Value-packed content
- Final tweet: CTA with engagement request

Number each tweet 1/10, 2/10, etc."""
            
        result = call_gemini(prompt, system_prompt, max_tokens=max_tokens)
        cleaned = clean_output(result)
        
        if not is_pro:
            cleaned += "\n\n--- \n💡 This is a Free Tier draft. Upgrade to PRO for full-length, professional-grade content."
            
        chunks = [cleaned[i:i+3800] for i in range(0, len(cleaned), 3800)]
        for chunk in chunks:
            await update.message.reply_text(chunk)
        
        del user_states[user_id]
        keyboard = [[InlineKeyboardButton("🔄 Generate Another", callback_data='generate')], [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("✅ Generation complete! What next?", reply_markup=reply_markup)

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
