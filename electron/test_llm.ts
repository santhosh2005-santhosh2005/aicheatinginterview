
import { LLMHelper } from "./LLMHelper";
import dotenv from "dotenv";
import path from "path";

// Load environment variables from the parent directory
dotenv.config({ path: path.join(__dirname, "..", ".env") });

async function runTest() {
    const useOllama = process.env.USE_OLLAMA === "true";
    const apiKey = process.env.GEMINI_API_KEY;

    if (!useOllama && !apiKey) {
        console.error("No API key or Ollama configuration found.");
        return;
    }

    console.log(`Testing with: ${useOllama ? "Ollama" : "Gemini"}`);

    let llmHelper;
    if (useOllama) {
        llmHelper = new LLMHelper(undefined, true, process.env.OLLAMA_MODEL, process.env.OLLAMA_URL);
    } else {
        llmHelper = new LLMHelper(apiKey, false);
    }

    const imagePath = path.join(__dirname, "..", "image.png");

    if (useOllama) {
        console.log("\n--- Checking available Ollama models ---");
        try {
            const models = await llmHelper.getOllamaModels();
            console.log("Available models:", models);
            const currentModel = llmHelper.getCurrentModel();
            console.log("Selected model:", currentModel);

            const visionKeywords = ['llava', 'moondream', 'vision', 'minicpm'];
            const isVision = visionKeywords.some(k => currentModel.toLowerCase().includes(k));

            if (!isVision) {
                console.warn("WARNING: Selected model does not appear to support vision capabilities.");
                console.warn("This is likely why answers are improper. Try installing 'llava' or 'moondream'.");
            } else {
                console.log("Model appears to support vision.");
            }
        } catch (error) {
            console.error("Error checking models:", error);
        }
    }

    console.log("\n--- Testing analyzeImageFile (Current behavior) ---");
    try {
        const result = await llmHelper.analyzeImageFile(imagePath);
        console.log("Result:", result.text);
    } catch (error) {
        console.error("Error in analyzeImageFile:", error);
    }

    console.log("\n--- Testing extractProblemFromImages (Desired behavior?) ---");
    try {
        const result = await llmHelper.extractProblemFromImages([imagePath]);
        console.log("Result:", JSON.stringify(result, null, 2));

        // If successful, try generating a solution
        console.log("\n--- Testing generateSolution ---");
        const solution = await llmHelper.generateSolution(result);
        console.log("Solution:", JSON.stringify(solution, null, 2));

    } catch (error) {
        console.error("Error in extractProblemFromImages/generateSolution:", error);
    }
}

runTest();
