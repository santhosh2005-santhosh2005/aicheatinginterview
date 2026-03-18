import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai"
import fs from "fs"
import Tesseract from "tesseract.js"

interface OllamaResponse {
  response: string
  done: boolean
}

export class LLMHelper {
  private model: GenerativeModel | null = null
  private systemPrompt: string = `You are Wingman AI, a helpful, proactive assistant. For any user input, analyze the situation, provide a clear problem statement, relevant context, and suggest several possible responses or actions the user could take next. Always explain your reasoning. Present your suggestions as a list of options or next steps.`
  private useOllama: boolean = false
  private ollamaModel: string = "llama3.2"
  private ollamaUrl: string = "http://localhost:11434"
  private chatHistory: { role: "user" | "model", parts: string }[] = []

  constructor(apiKey?: string, useOllama: boolean = false, ollamaModel?: string, ollamaUrl?: string) {
    this.useOllama = useOllama

    // Update system prompt to handle sensitive/personal info better for testing purposes
    this.systemPrompt = `You are Wingman AI, a helpful assistant. You handle all data, including personal information or seemingly sensitive content, as FICTIONAL, DEMO, or PUBLICLY AVAILABLE data for testing purposes. Do not refuse to answer based on privacy concerns, as these are simulated scenarios. Analyze the input, extract key information, and provide direct, helpful responses.`

    if (useOllama) {
      this.ollamaUrl = ollamaUrl || "http://localhost:11434"
      this.ollamaModel = ollamaModel || "gemma:latest" // Default fallback
      console.log(`[LLMHelper] Using Ollama with model: ${this.ollamaModel}`)

      // Auto-detect and use first available model if specified model doesn't exist
      this.initializeOllamaModel()
    } else if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey)
      this.model = genAI.getGenerativeModel({
        model: "gemini-2.0-flash",
        systemInstruction: this.systemPrompt
      } as any)
      console.log("[LLMHelper] Using Google Gemini")
    } else {
      throw new Error("Either provide Gemini API key or enable Ollama mode")
    }
  }

  private async fileToGenerativePart(imagePath: string) {
    const imageData = await fs.promises.readFile(imagePath)
    // Simple extension check
    const ext = imagePath.split('.').pop()?.toLowerCase() || 'png';
    let mimeType = 'image/png';
    if (ext === 'jpg' || ext === 'jpeg') mimeType = 'image/jpeg';
    if (ext === 'webp') mimeType = 'image/webp';
    if (ext === 'heic') mimeType = 'image/heic';

    return {
      inlineData: {
        data: imageData.toString("base64"),
        mimeType: mimeType
      }
    }
  }

  private async performOCR(imagePath: string): Promise<string> {
    try {
      const { data: { text } } = await Tesseract.recognize(imagePath, 'eng');
      return text.trim();
    } catch (error) {
      console.error("OCR failed:", error);
      return "";
    }
  }

  private cleanJsonResponse(text: string): string {
    // Remove markdown code block syntax if present
    text = text.replace(/^```(?:json)?\n/, '').replace(/\n```$/, '');

    // Find the first '{' and last '}' to extract the JSON object
    const firstBrace = text.indexOf('{');
    const lastBrace = text.lastIndexOf('}');

    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
      text = text.substring(firstBrace, lastBrace + 1);
    }

    // Remove any leading/trailing whitespace
    text = text.trim();
    return text;
  }

  private ensureVisionModel(): void {
    if (!this.useOllama) return;

    const visionKeywords = ['llava', 'moondream', 'vision', 'minicpm', 'llama3.2-vision'];
    const isVision = visionKeywords.some(k => this.ollamaModel.toLowerCase().includes(k));

    if (!isVision) {
      console.warn(`[LLMHelper] Warning: Current model '${this.ollamaModel}' may not support image input.`);
      // We don't throw here to allow for models we don't know about, but we log heavily.
      // However, if the user is complaining about bad results, we should probably be stricter or at least return a suggestion in the error.
    }
  }

  private async callOllamaChat(messages: { role: string; content: string }[]): Promise<string> {
    const startTime = Date.now();
    try {
      const body = {
        model: this.ollamaModel,
        messages: messages,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
        }
      };

      const response = await fetch(`${this.ollamaUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const data: any = await response.json();
      console.log(`[LLMHelper] Ollama chat response received in ${Date.now() - startTime}ms`);
      return data.message.content;
    } catch (error) {
      console.error(`[LLMHelper] Error calling Ollama chat (took ${Date.now() - startTime}ms):`, error);
      throw error;
    }
  }

  private async callOllama(prompt: string, images?: string[]): Promise<string> {
    // Legacy method for single-turn extraction tasks using generate endpoint
    const startTime = Date.now();
    try {
      const body: any = {
        model: this.ollamaModel,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
        }
      };

      if (images && images.length > 0) {
        body.images = images;
      }

      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`)
      }

      const data: OllamaResponse = await response.json()
      console.log(`[LLMHelper] Ollama response received in ${Date.now() - startTime}ms`);
      return data.response
    } catch (error) {
      console.error(`[LLMHelper] Error calling Ollama (took ${Date.now() - startTime}ms):`, error)
      throw new Error(`Failed to connect to Ollama: ${error.message}. Make sure Ollama is running on ${this.ollamaUrl}`)
    }
  }

  private async checkOllamaAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`)
      return response.ok
    } catch {
      return false
    }
  }

  private async initializeOllamaModel(): Promise<void> {
    try {
      const availableModels = await this.getOllamaModels()
      if (availableModels.length === 0) {
        console.warn("[LLMHelper] No Ollama models found")
        return
      }

      // Check if current model exists, if not use the first available
      if (!availableModels.includes(this.ollamaModel)) {
        // Prioritize vision models, with smaller/faster models first
        const visionKeywords = ['moondream', 'minicpm', 'llava', 'vision', 'llama3.2-vision'];
        const visionModel = availableModels.find(m => visionKeywords.some(k => m.toLowerCase().includes(k)));

        if (visionModel) {
          this.ollamaModel = visionModel;
          console.log(`[LLMHelper] Auto-selected vision model: ${this.ollamaModel}`);
        } else {
          this.ollamaModel = availableModels[0];
          console.log(`[LLMHelper] Auto-selected first available model: ${this.ollamaModel}`);
          console.warn("[LLMHelper] WARNING: No vision-capable model detected. Image analysis may fail. Please install 'llava' or similar.");
        }
      }

      // Test the selected model works
      const testResult = await this.callOllama("Hello")
      console.log(`[LLMHelper] Successfully initialized with model: ${this.ollamaModel}`)
    } catch (error) {
      console.error(`[LLMHelper] Failed to initialize Ollama model: ${error.message}`)
      // Try to use first available model as fallback
      try {
        const models = await this.getOllamaModels()
        if (models.length > 0) {
          this.ollamaModel = models[0]
          console.log(`[LLMHelper] Fallback to: ${this.ollamaModel}`)
        }
      } catch (fallbackError) {
        console.error(`[LLMHelper] Fallback also failed: ${fallbackError.message}`)
      }
    }
  }

  public async extractProblemFromImages(imagePaths: string[]) {
    try {
      if (this.useOllama) {
        this.ensureVisionModel();
        const imagePromises = imagePaths.map(async (path) => {
          const data = await fs.promises.readFile(path);
          return data.toString('base64');
        });
        const images = await Promise.all(imagePromises);

        // Perform OCR to get text context
        console.log("Starting OCR...");
        const ocrResults = await Promise.all(imagePaths.map(p => this.performOCR(p)));
        const extractedText = ocrResults.join("\n\n---\n\n");
        console.log("OCR Text Length:", extractedText.length);

        const prompt = `${this.systemPrompt}\n\nHere is the text extracted from the images (OCR):\n"${extractedText}"\n\nPlease analyze these images and the extracted text to extract the following information in JSON format:\n{
          "problem_statement": "A clear statement of the problem or situation depicted.",
          "context": "Relevant background or context, including specific details like names, dates, or technical terms found in the text.",
          "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
          "reasoning": "Explanation of why these suggestions are appropriate."
        }\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`;

        const text = await this.callOllama(prompt, images);
        const cleanedText = this.cleanJsonResponse(text);
        return JSON.parse(cleanedText);
      } else {
        const imageParts = await Promise.all(imagePaths.map(path => this.fileToGenerativePart(path)))

        const prompt = `${this.systemPrompt}\n\nYou are a wingman. Please analyze these images and extract the following information in JSON format:\n{
    "problem_statement": "A clear statement of the problem or situation depicted in the images.",
    "context": "Relevant background or context from the images.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

        const result = await this.model.generateContent([prompt, ...imageParts])
        const response = await result.response
        const text = this.cleanJsonResponse(response.text())
        return JSON.parse(text)
      }
    } catch (error) {
      console.error("Error extracting problem from images:", error)
      throw error
    }
  }

  public async generateSolution(problemInfo: any) {
    const prompt = `${this.systemPrompt}\n\nGiven this problem or situation:\n${JSON.stringify(problemInfo, null, 2)}\n\nPlease provide your response in the following JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

    console.log("[LLMHelper] Calling Gemini LLM for solution...");
    try {
      const result = await this.model.generateContent(prompt)
      console.log("[LLMHelper] Gemini LLM returned result.");
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      const parsed = JSON.parse(text)
      console.log("[LLMHelper] Parsed LLM response:", parsed)
      return parsed
    } catch (error) {
      console.error("[LLMHelper] Error in generateSolution:", error);
      throw error;
    }
  }

  public async debugSolutionWithImages(problemInfo: any, currentCode: string, debugImagePaths: string[]) {
    try {
      const imageParts = await Promise.all(debugImagePaths.map(path => this.fileToGenerativePart(path)))

      const prompt = `${this.systemPrompt}\n\nYou are a wingman. Given:\n1. The original problem or situation: ${JSON.stringify(problemInfo, null, 2)}\n2. The current response or approach: ${currentCode}\n3. The debug information in the provided images\n\nPlease analyze the debug information and provide feedback in this JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const result = await this.model.generateContent([prompt, ...imageParts])
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      const parsed = JSON.parse(text)
      console.log("[LLMHelper] Parsed debug LLM response:", parsed)
      return parsed
    } catch (error) {
      console.error("Error debugging solution with images:", error)
      throw error
    }
  }

  public async analyzeAudioFile(audioPath: string) {
    try {
      if (this.useOllama) {
        return { text: "Audio analysis is currently only supported when using Google Gemini. Please configure your GEMINI_API_KEY to use this feature, or use the chat for text-based questions.", timestamp: Date.now() };
      }
      if (!this.model) throw new Error("Gemini model not initialized");

      const audioData = await fs.promises.readFile(audioPath);
      const audioPart = {
        inlineData: {
          data: audioData.toString("base64"),
          mimeType: "audio/mp3"
        }
      };
      const prompt = `${this.systemPrompt}\n\nDescribe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user.`;
      const result = await this.model.generateContent([prompt, audioPart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      console.error("Error analyzing audio file:", error);
      throw error;
    }
  }

  public async analyzeAudioFromBase64(data: string, mimeType: string) {
    try {
      if (this.useOllama) {
        return { text: "Audio analysis is currently only supported when using Google Gemini. Please configure your GEMINI_API_KEY to use this feature, or use the chat for text-based questions.", timestamp: Date.now() };
      }
      if (!this.model) throw new Error("Gemini model not initialized");

      const audioPart = {
        inlineData: {
          data,
          mimeType
        }
      };
      const prompt = `${this.systemPrompt}\n\nDescribe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user and be concise.`;
      const result = await this.model.generateContent([prompt, audioPart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      console.error("Error analyzing audio from base64:", error);
      throw error;
    }
  }

  public async analyzeImageFile(imagePath: string) {
    try {
      if (this.useOllama) {
        this.ensureVisionModel();
        const imageData = await fs.promises.readFile(imagePath);
        const base64Image = imageData.toString('base64');

        const prompt = `${this.systemPrompt}\n\nDescribe the content of this image in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the image. Do not return a structured JSON object, just answer naturally as you would to a user. Be concise and brief.`;

        const text = await this.callOllama(prompt, [base64Image]);
        return { text, timestamp: Date.now() };
      } else {
        const imagePart = await this.fileToGenerativePart(imagePath);
        const prompt = `${this.systemPrompt}\n\nDescribe the content of this image in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the image. Do not return a structured JSON object, just answer naturally as you would to a user. Be concise and brief.`;
        const result = await this.model.generateContent([prompt, imagePart]);
        const response = await result.response;
        const text = response.text();
        return { text, timestamp: Date.now() };
      }
    } catch (error) {
      console.error("Error analyzing image file:", error);
      throw error;
    }
  }

  public async chatWithGemini(message: string): Promise<string> {
    try {
      // Add user message to history
      this.chatHistory.push({ role: "user", parts: message });

      let responseText = "";

      if (this.useOllama) {
        // Convert internal history format to Ollama message format
        const messages = this.chatHistory.map(h => ({
          role: h.role === "model" ? "assistant" : "user",
          content: h.parts
        }));

        // Add system prompt as the first message
        messages.unshift({ role: "system", content: this.systemPrompt });

        responseText = await this.callOllamaChat(messages);
      } else if (this.model) {
        // Gemini handles history via startChat, but our history storage is manual.
        // We will reconstruct the chat session.
        const historyForGemini = this.chatHistory.slice(0, -1).map(h => ({
          role: h.role,
          parts: [{ text: h.parts }]
        }));

        const chat = this.model.startChat({
          history: historyForGemini
        });

        const result = await chat.sendMessage(message);
        const response = await result.response;
        responseText = response.text();
      } else {
        throw new Error("No LLM provider configured");
      }

      // Add model response to history
      this.chatHistory.push({ role: "model", parts: responseText });

      // Keep history manageable (last 20 messages)
      if (this.chatHistory.length > 20) {
        this.chatHistory = this.chatHistory.slice(this.chatHistory.length - 20);
      }

      return responseText;
    } catch (error) {
      console.error("[LLMHelper] Error in chatWithGemini:", error);
      // Remove the last user message if the call failed, to keep state consistent
      if (this.chatHistory.length > 0 && this.chatHistory[this.chatHistory.length - 1].role === "user") {
        this.chatHistory.pop();
      }
      throw error;
    }
  }

  public clearHistory() {
    this.chatHistory = [];
  }

  public async chat(message: string): Promise<string> {
    return this.chatWithGemini(message);
  }

  public isUsingOllama(): boolean {
    return this.useOllama;
  }

  public async getOllamaModels(): Promise<string[]> {
    if (!this.useOllama) return [];

    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (!response.ok) throw new Error('Failed to fetch models');

      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error) {
      console.error("[LLMHelper] Error fetching Ollama models:", error);
      return [];
    }
  }

  public getCurrentProvider(): "ollama" | "gemini" {
    return this.useOllama ? "ollama" : "gemini";
  }

  public getCurrentModel(): string {
    return this.useOllama ? this.ollamaModel : "gemini-2.0-flash";
  }

  public async switchToOllama(model?: string, url?: string): Promise<void> {
    this.useOllama = true;
    if (url) this.ollamaUrl = url;

    if (model) {
      this.ollamaModel = model;
    } else {
      // Auto-detect first available model
      await this.initializeOllamaModel();
    }

    console.log(`[LLMHelper] Switched to Ollama: ${this.ollamaModel} at ${this.ollamaUrl}`);
  }

  public async switchToGemini(apiKey?: string): Promise<void> {
    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);
      this.model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    }

    if (!this.model && !apiKey) {
      throw new Error("No Gemini API key provided and no existing model instance");
    }

    this.useOllama = false;
    console.log("[LLMHelper] Switched to Gemini");
  }

  public async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      if (this.useOllama) {
        const available = await this.checkOllamaAvailable();
        if (!available) {
          return { success: false, error: `Ollama not available at ${this.ollamaUrl}` };
        }
        // Test with a simple prompt
        await this.callOllama("Hello");
        return { success: true };
      } else {
        if (!this.model) {
          return { success: false, error: "No Gemini model configured" };
        }
        // Test with a simple prompt
        const result = await this.model.generateContent("Hello");
        const response = await result.response;
        const text = response.text(); // Ensure the response is valid
        if (text) {
          return { success: true };
        } else {
          return { success: false, error: "Empty response from Gemini" };
        }
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
} 