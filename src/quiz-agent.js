const fs = require('fs');
const path = require('path');

class QuizAgent {
  constructor() {
    this.difficultyLevels = ['Beginner', 'Normal', 'High'];
    this.score = 0;
    this.totalQuestions = 0;
    this.currentTopic = null;
    this.quizData = [];
  }

  /**
   * Extracts content from markdown files in the docs directory
   */
  extractContentFromDocs(docsDirPath) {
    const topics = {};
    
    const walkSync = (dir, filelist = []) => {
      const files = fs.readdirSync(dir);
      
      files.forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        
        if (stat.isDirectory()) {
          filelist = walkSync(filePath, filelist);
        } else if (path.extname(file) === '.md') {
          filelist.push(filePath);
        }
      });
      
      return filelist;
    };
    
    const markdownFiles = walkSync(docsDirPath);
    
    markdownFiles.forEach(filePath => {
      const content = fs.readFileSync(filePath, 'utf-8');
      const fileName = path.basename(filePath, '.md');
      const relativePath = path.relative(docsDirPath, filePath).replace(/\\/g, '/');
      
      // Extract sections from the content
      const sections = this.extractSections(content, fileName);
      
      // Group content by topic (directory name)
      const topicName = path.dirname(relativePath).split('/')[0];
      if (!topics[topicName]) {
        topics[topicName] = [];
      }
      
      topics[topicName].push(...sections);
    });
    
    return topics;
  }

  /**
   * Extract sections from markdown content
   */
  extractSections(content, fileName) {
    const sections = [];
    const lines = content.split('\n');
    let currentSection = { title: fileName.replace(/-/g, ' '), content: [] };
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      if (line.startsWith('# ')) {
        // New main heading
        if (currentSection.title !== fileName.replace(/-/g, ' ')) {
          sections.push({ ...currentSection });
        }
        currentSection = { title: line.substring(2), content: [] };
      } else if (line.startsWith('## ')) {
        // New subsection
        if (currentSection.title !== fileName.replace(/-/g, ' ') || currentSection.content.length > 0) {
          sections.push({ ...currentSection });
        }
        currentSection = { title: line.substring(3), content: [] };
      } else if (line !== '') {
        currentSection.content.push(line);
      }
    }
    
    if (currentSection.title !== fileName.replace(/-/g, ' ') || currentSection.content.length > 0) {
      sections.push({ ...currentSection });
    }
    
    return sections;
  }

  /**
   * Generates questions based on content and difficulty level
   */
  generateQuestions(sectionContent, difficulty) {
    // This is a simplified question generation logic
    // In a real implementation, you might use AI or more sophisticated NLP
    const contentText = sectionContent.content.join(' ');
    const sentences = contentText.split(/[.!?]+/).filter(sentence => sentence.trim().length > 10);
    
    const questions = [];
    
    switch(difficulty) {
      case 'Beginner':
        // Simple factual questions
        for (let i = 0; i < Math.min(2, sentences.length); i++) {
          const sentence = sentences[i];
          if (sentence.includes(':') || sentence.includes('is') || sentence.includes('are')) {
            const parts = sentence.split(':');
            if (parts.length > 1) {
              questions.push({
                question: `What is ${parts[0].trim().toLowerCase()}?`,
                options: [`Option A`, `Option B`, sentence.trim(), `Option D`],
                correctAnswer: sentence.trim(),
                explanation: `Based on the text: "${sentence.trim()}"`
              });
            }
          }
        }
        break;
        
      case 'Normal':
        // More complex questions
        for (let i = 0; i < Math.min(3, sentences.length); i++) {
          if (i % 2 === 0) {
            const sentence = sentences[i];
            const fakeOptions = [
              sentence.replace(/\w+/, 'FAKE'),
              sentence.replace(/the/, 'a'),
              sentence,
              sentence.replace(/\s+/g, ' ')
            ];
            
            questions.push({
              question: `Which of the following is true about ${sectionContent.title.toLowerCase()}?`,
              options: this.shuffleArray(fakeOptions),
              correctAnswer: sentence,
              explanation: `According to the text: "${sentence}"`
            });
          }
        }
        break;
        
      case 'High':
        // Complex analytical questions
        for (let i = 0; i < Math.min(4, sentences.length); i++) {
          if (i % 3 === 0) {
            const sentence = sentences[i];
            const fakeOptions = [
              `This is definitely incorrect`,
              `This is not quite right`,
              sentence,
              `This is false for sure`
            ];
            
            questions.push({
              question: `Based on the principles discussed in ${sectionContent.title.toLowerCase()}, which statement applies?`,
              options: this.shuffleArray(fakeOptions),
              correctAnswer: sentence,
              explanation: `The text states: "${sentence}"`
            });
          }
        }
        break;
    }
    
    // Shuffle the questions
    return this.shuffleArray(questions);
  }

  /**
   * Shuffles an array using Fisher-Yates algorithm
   */
  shuffleArray(array) {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
    }
    return newArray;
  }

  /**
   * Starts the quiz for a specific topic
   */
  startQuizForTopic(topicName, topicSections) {
    console.log(`\\n========================================`);
    console.log(`Starting quiz for topic: ${topicName}`);
    console.log(`========================================\\n`);
    
    this.currentTopic = topicName;
    this.score = 0;
    this.totalQuestions = 0;
    
    this.difficultyLevels.forEach((difficulty, index) => {
      console.log(`\\n--- ${difficulty} Level Questions ---`);
      
      // Use the first section for now, but in practice you might use multiple sections
      const sectionContent = topicSections[0] || { title: topicName, content: ['General content'] };
      const questions = this.generateQuestions(sectionContent, difficulty);
      
      questions.forEach((q, qIndex) => {
        this.askQuestion(q, qIndex + 1, questions.length);
      });
    });
    
    this.showSummary();
  }

  /**
   * Asks a single question and handles the response
   */
  askQuestion(questionObj, questionNumber, totalInLevel) {
    console.log(`\\nQuestion ${questionNumber}/${totalInLevel}:`);
    console.log(questionObj.question);
    
    questionObj.options.forEach((option, idx) => {
      console.log(`${String.fromCharCode(65 + idx)}) ${option}`);
    });
    
    // Simulated user input - in a real implementation, you'd get actual input
    const userInput = this.simulateUserInput(); // This simulates user input
    
    this.totalQuestions++;
    
    if (questionObj.options[userInput - 1] === questionObj.correctAnswer) {
      console.log('✅ Correct!');
      this.score++;
    } else {
      console.log('❌ Incorrect.');
      console.log(`Correct answer: ${questionObj.correctAnswer}`);
    }
    
    console.log(`Explanation: ${questionObj.explanation}\\n`);
  }

  /**
   * Simulates user input (for demonstration purposes)
   */
  simulateUserInput() {
    // In a real implementation, you would get actual user input
    // For demo purposes, we'll randomly select an option
    return Math.floor(Math.random() * 4) + 1; // Return 1-4 representing A-D
  }

  /**
   * Shows a summary of the quiz results
   */
  showSummary() {
    console.log(`\\n========================================`);
    console.log(`Quiz Summary for Topic: ${this.currentTopic}`);
    console.log(`========================================`);
    console.log(`Score: ${this.score}/${this.totalQuestions}`);
    console.log(`Percentage: ${this.totalQuestions ? Math.round((this.score / this.totalQuestions) * 100) : 0}%`);
    
    if (this.score === this.totalQuestions) {
      console.log('🎉 Perfect score! Excellent job!');
    } else if (this.score >= this.totalQuestions * 0.7) {
      console.log('👍 Good job! You have a solid understanding.');
    } else {
      console.log('📚 Keep practicing to improve your knowledge.');
    }
    console.log(`========================================\\n`);
  }

  /**
   * Runs the full quiz agent
   */
  run() {
    console.log("🤖 AI Textbook Quiz Agent");
    console.log("Generating quizzes based on AI textbook content...");
    
    const topics = this.extractContentFromDocs('./docs');
    
    // For this demo, we'll run quizzes for all topics
    Object.entries(topics).forEach(([topicName, topicSections]) => {
      this.startQuizForTopic(topicName, topicSections);
    });
  }
}

// Initialize and run the quiz agent
const quizAgent = new QuizAgent();
quizAgent.run();

module.exports = QuizAgent;