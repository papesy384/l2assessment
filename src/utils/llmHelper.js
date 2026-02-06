import Groq from 'groq-sdk';

/**
 * LLM Helper for categorizing customer support messages
 * Using Groq API for AI-powered categorization
 */

// Initialize Groq client
const groq = new Groq({
  apiKey: import.meta.env.VITE_GROQ_API_KEY,
  dangerouslyAllowBrowser: true // Required for browser-based calls (not recommended for production!)
});

/**
 * Categorize a customer support message using Groq AI
 * Returns JSON with category, sentiment (Neutral/Angry), and priority_score (1-5).
 *
 * @param {string} message - The customer support message
 * @returns {Promise<{category: string, sentiment: string, priority_score: number, reasoning: string}>}
 */
export async function categorizeMessage(message) {
  try {
    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "user",
          content: `Analyze this customer support message and respond with ONLY a valid JSON object (no markdown, no code fences, no extra text).

Required JSON shape:
{
  "category": "one of: Billing Issue, Technical Problem, Feature Request, General Inquiry, Unknown",
  "sentiment": "Neutral or Angry",
  "priority_score": number from 1 to 5 (1=lowest, 5=highest priority),
  "reasoning": "brief explanation of your classification"
}

Customer message:
${message}`
        }
      ],
      temperature: 0.3,
    });

    const content = response.choices[0].message.content.trim();
    const parsed = parseJsonResponse(content);
    return normalizeCategorizationResult(parsed);
  } catch (error) {
    console.warn('Groq API failed, using mock response:', error.message);
    return getMockCategorization(message);
  }
}

/**
 * Parse JSON from LLM response (handles markdown code blocks if present)
 * @param {string} content - Raw response content
 * @returns {{ category?: string, sentiment?: string, priority_score?: number, reasoning?: string }}
 */
function parseJsonResponse(content) {
  let jsonStr = content;
  const codeBlockMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    jsonStr = codeBlockMatch[1].trim();
  }
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    console.warn('Failed to parse LLM JSON, using fallback:', e.message);
    return {};
  }
}

/**
 * Ensure result has required fields with valid types and ranges
 */
function normalizeCategorizationResult(parsed) {
  const category = typeof parsed.category === 'string' && parsed.category.trim()
    ? parsed.category.trim()
    : 'Unknown';
  const sentiment = parsed.sentiment === 'Angry' ? 'Angry' : 'Neutral';
  let priority_score = Number(parsed.priority_score);
  if (!Number.isInteger(priority_score) || priority_score < 1 || priority_score > 5) {
    priority_score = 3;
  }
  const reasoning = typeof parsed.reasoning === 'string' && parsed.reasoning.trim()
    ? parsed.reasoning.trim()
    : `Category: ${category}, Sentiment: ${sentiment}, Priority: ${priority_score}.`;
  return {
    category,
    sentiment,
    priority_score,
    reasoning,
  };
}

/**
 * Mock categorization for when API is unavailable.
 * Returns same shape as API: category, sentiment (Neutral/Angry), priority_score (1-5), reasoning.
 */
function getMockCategorization(message) {
  const lowerMessage = message.toLowerCase();

  // Detect angry sentiment for priority/sentiment
  const angryIndicators = ['angry', 'furious', 'outraged', 'unacceptable', 'worst', 'terrible', 'horrible', '!!!', 'urgent', 'asap', 'immediately'];
  const isAngry = angryIndicators.some(w => lowerMessage.includes(w)) || (message.match(/!/g) || []).length >= 2;
  const sentiment = isAngry ? 'Angry' : 'Neutral';
  const basePriority = isAngry ? 4 : 2;

  const reasoningVariations = {
    billing: "Based on keywords related to payments and billing, this appears to be a billing-related inquiry.",
    technical: "This message describes technical difficulties or system errors that may require engineering review.",
    feature: "The customer is requesting enhancements or new functionality.",
    inquiry: "This appears to be a general question about the product or service.",
    positive: "The customer is expressing satisfaction or gratitude.",
    ambiguous: "The message doesn't contain clear indicators for automatic categorization. Manual review recommended.",
  };

  // Billing-related
  if (lowerMessage.includes('bill') || lowerMessage.includes('payment') || lowerMessage.includes('charge') ||
      lowerMessage.includes('invoice') || lowerMessage.includes('subscription') || lowerMessage.includes('refund')) {
    return {
      category: "Billing Issue",
      sentiment,
      priority_score: Math.min(5, basePriority + 1),
      reasoning: reasoningVariations.billing,
    };
  }

  // Technical problem
  if (lowerMessage.includes('bug') || lowerMessage.includes('error') || lowerMessage.includes('broken') ||
      lowerMessage.includes('not working') || lowerMessage.includes('crash') || lowerMessage.includes('down') ||
      lowerMessage.includes('server') || lowerMessage.includes('loading') || lowerMessage.includes('issue')) {
    return {
      category: "Technical Problem",
      sentiment,
      priority_score: isAngry ? 5 : Math.min(5, basePriority + 2),
      reasoning: reasoningVariations.technical,
    };
  }

  // Feature request
  if (lowerMessage.includes('feature') || lowerMessage.includes('improve') || lowerMessage.includes('suggestion') ||
      lowerMessage.includes('would like to see') || lowerMessage.includes('enhancement')) {
    return {
      category: "Feature Request",
      sentiment,
      priority_score: basePriority,
      reasoning: reasoningVariations.feature,
    };
  }

  // Positive feedback
  if ((lowerMessage.includes('thank') || lowerMessage.includes('thanks') || lowerMessage.includes('appreciate')) &&
      !lowerMessage.includes('but') && !lowerMessage.includes('however')) {
    return {
      category: "General Inquiry",
      sentiment: "Neutral",
      priority_score: 1,
      reasoning: reasoningVariations.positive,
    };
  }

  // General inquiry
  if (lowerMessage.includes('how') || lowerMessage.includes('what') || lowerMessage.includes('?') ||
      lowerMessage.includes('can i') || lowerMessage.includes('is there')) {
    return {
      category: "General Inquiry",
      sentiment,
      priority_score: basePriority,
      reasoning: reasoningVariations.inquiry,
    };
  }

  return {
    category: "General Inquiry",
    sentiment,
    priority_score: basePriority,
    reasoning: reasoningVariations.ambiguous,
  };
}
