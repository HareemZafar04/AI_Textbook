import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import clsx from 'clsx';

const QuizAgent = () => {
  const [currentTopic, setCurrentTopic] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [selectedOption, setSelectedOption] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [difficulty, setDifficulty] = useState('Beginner'); // 'Beginner', 'Normal', 'High'
  const [topics] = useState([
    { 
      id: 'foundations', 
      name: 'Foundations of AI',
      content: [
        {
          title: 'History of AI',
          text: 'Artificial Intelligence (AI) has roots dating back to ancient times with myths of artificial beings. The term "Artificial Intelligence" was coined in 1956 by John McCarthy at the Dartmouth Conference. Early AI research focused on solving specific problems like playing chess or proving mathematical theorems.'
        },
        {
          title: 'Core Concepts',
          text: 'AI encompasses machine learning, deep learning, neural networks, natural language processing, computer vision, and robotics. Machine learning allows systems to learn and improve from experience without explicit programming.'
        }
      ]
    },
    { 
      id: 'ml', 
      name: 'Machine Learning',
      content: [
        {
          title: 'Introduction to ML',
          text: 'Machine Learning is a subset of AI that enables computers to learn and make decisions from data without being explicitly programmed. The three main types are supervised, unsupervised, and reinforcement learning.'
        },
        {
          title: 'Supervised Learning',
          text: 'In supervised learning, the model learns from labeled training data. The goal is to map inputs to outputs based on example input-output pairs. Common algorithms include linear regression, logistic regression, and support vector machines.'
        }
      ]
    },
    { 
      id: 'nlp', 
      name: 'Natural Language Processing',
      content: [
        {
          title: 'Introduction to NLP',
          text: 'Natural Language Processing (NLP) is a field of AI focused on making computers understand, interpret, and generate human language. NLP combines computational linguistics with machine learning to process and analyze large amounts of natural language data.'
        }
      ]
    }
  ]);

  // Sample quiz questions based on topics
  const [allQuestions, setAllQuestions] = useState([]);

  // Function to generate questions based on content
  const generateQuestions = (topicContent, level) => {
    const questions = [];
    let numQuestions = 0;
    
    switch(level) {
      case 'Beginner':
        numQuestions = 3;
        break;
      case 'Normal':
        numQuestions = 5;
        break;
      case 'High':
        numQuestions = 7;
        break;
      default:
        numQuestions = 3;
    }
    
    topicContent.forEach(section => {
      for (let i = 0; i < numQuestions; i++) {
        // Extract key phrases from the content to form questions
        const words = section.text.split(' ');
        const randomIdx = Math.floor(Math.random() * (words.length - 5));
        const questionPhrase = words.slice(randomIdx, randomIdx + 3).join(' ');
        
        questions.push({
          id: `${section.title}-${i}`,
          question: `What is the relationship between "${questionPhrase}" and ${section.title}?`,
          options: [
            section.text.split('.')[0],
            `Another option about ${section.title}`,
            `Different aspect of ${section.title}`,
            `Final choice regarding ${section.title}`
          ],
          correctAnswer: section.text.split('.')[0],
          explanation: `Based on the content in "${section.title}": ${section.text}`
        });
      }
    });
    
    return questions;
  };

  const startQuiz = (topicId, level) => {
    const topic = topics.find(t => t.id === topicId);
    setCurrentTopic(topic);
    setDifficulty(level);
    const questions = generateQuestions(topic.content, level);
    setAllQuestions(questions);
    setCurrentQuestion(0);
    setScore(0);
    setSelectedOption('');
    setShowFeedback(false);
    setQuizCompleted(false);
  };

  const handleSelectOption = (option) => {
    if (showFeedback) return; // Prevent selecting option after answer is shown
    setSelectedOption(option);
  };

  const handleSubmitAnswer = () => {
    if (!selectedOption) return;
    
    const isCorrect = selectedOption === allQuestions[currentQuestion].correctAnswer;
    setShowFeedback(true);
    
    if (isCorrect) {
      setScore(score + 1);
    }
    
    setTimeout(() => {
      if (currentQuestion < allQuestions.length - 1) {
        // Move to next question
        setCurrentQuestion(currentQuestion + 1);
        setSelectedOption('');
        setShowFeedback(false);
      } else {
        // Quiz completed
        setQuizCompleted(true);
      }
    }, 2000);
  };

  const resetQuiz = () => {
    setCurrentTopic(null);
    setQuizCompleted(false);
    setAllQuestions([]);
  };

  const renderDifficultyButtons = (topicId) => (
    <div className="flex flex-wrap gap-3 mt-3">
      {['Beginner', 'Normal', 'High'].map((level) => (
        <button
          key={level}
          onClick={() => startQuiz(topicId, level)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            difficulty === level
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
          }`}
        >
          {level}
        </button>
      ))}
    </div>
  );

  return (
    <Layout title="AI Textbook Quiz Agent" description="Interactive quizzes based on AI textbook content">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">🤖 AI Textbook Quiz Agent</h1>
          <p className="text-xl text-gray-600">
            Test your knowledge with quizzes based on the content of this AI textbook
          </p>
        </header>

        {!currentTopic && !quizCompleted && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-center mb-6">Select a Topic</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {topics.map((topic) => (
                <div 
                  key={topic.id} 
                  className="border border-gray-200 rounded-xl p-6 bg-white shadow-sm hover:shadow-md transition-shadow"
                >
                  <h3 className="text-xl font-bold mb-3">{topic.name}</h3>
                  <p className="text-gray-600 mb-4">
                    {topic.content.length} section{topic.content.length > 1 ? 's' : ''} available
                  </p>
                  <button
                    onClick={() => startQuiz(topic.id, 'Beginner')}
                    className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Start Quiz
                  </button>
                  {renderDifficultyButtons(topic.id)}
                </div>
              ))}
            </div>
          </div>
        )}

        {currentTopic && !quizCompleted && (
          <div className="max-w-2xl mx-auto">
            <div className="mb-6 bg-blue-50 p-4 rounded-lg">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">{currentTopic.name}</h2>
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                  {difficulty} Level
                </span>
              </div>
              <div className="mt-3 w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{ width: `${((currentQuestion + 1) / allQuestions.length) * 100}%` }}
                ></div>
              </div>
              <div className="text-right text-sm text-gray-600 mt-1">
                Question {currentQuestion + 1} of {allQuestions.length}
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold mb-6">
                {allQuestions[currentQuestion]?.question}
              </h3>
              
              <div className="space-y-3 mb-6">
                {allQuestions[currentQuestion]?.options.map((option, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSelectOption(option)}
                    disabled={showFeedback}
                    className={`w-full text-left p-4 rounded-lg border transition-colors ${
                      selectedOption === option
                        ? option === allQuestions[currentQuestion]?.correctAnswer
                          ? 'border-green-500 bg-green-50'
                          : 'border-red-500 bg-red-50'
                        : showFeedback
                          ? option === allQuestions[currentQuestion]?.correctAnswer
                            ? 'border-green-500 bg-green-50'
                            : 'border-gray-200 bg-gray-50'
                          : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
                    }`}
                  >
                    <span className="font-medium mr-2">
                      {String.fromCharCode(65 + idx)})
                    </span>
                    {option}
                  </button>
                ))}
              </div>
              
              {showFeedback && (
                <div className={`p-4 rounded-lg mb-4 ${
                  selectedOption === allQuestions[currentQuestion]?.correctAnswer
                    ? 'bg-green-50 border border-green-200'
                    : 'bg-red-50 border border-red-200'
                }`}>
                  <p className={`font-medium ${
                    selectedOption === allQuestions[currentQuestion]?.correctAnswer
                      ? 'text-green-800'
                      : 'text-red-800'
                  }`}>
                    {selectedOption === allQuestions[currentQuestion]?.correctAnswer
                      ? '✅ Correct!'
                      : '❌ Incorrect.'}
                  </p>
                  <p className="mt-2 text-sm text-gray-700">
                    {allQuestions[currentQuestion]?.explanation}
                  </p>
                </div>
              )}
              
              <div className="flex justify-between items-center">
                <div className="text-lg font-semibold">
                  Score: <span className="text-blue-600">{score}</span> / {currentQuestion + 1}
                </div>
                <button
                  onClick={handleSubmitAnswer}
                  disabled={!selectedOption || showFeedback}
                  className={`py-2 px-6 rounded-lg font-medium ${
                    selectedOption && !showFeedback
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {currentQuestion < allQuestions.length - 1 ? 'Next Question' : 'Finish Quiz'}
                </button>
              </div>
            </div>
          </div>
        )}

        {quizCompleted && (
          <div className="max-w-2xl mx-auto text-center">
            <div className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm">
              <h2 className="text-2xl font-bold mb-4">Quiz Completed!</h2>
              <div className="text-5xl font-bold text-blue-600 mb-4">
                {score}/{allQuestions.length}
              </div>
              <div className="text-xl mb-2">
                Your Score: {Math.round((score / allQuestions.length) * 100)}%
              </div>
              
              {score === allQuestions.length ? (
                <div className="text-green-600 text-lg font-medium mb-6">🎉 Perfect Score! Excellent Job!</div>
              ) : score >= allQuestions.length * 0.7 ? (
                <div className="text-blue-600 text-lg font-medium mb-6">👍 Good Job! You have a solid understanding.</div>
              ) : (
                <div className="text-yellow-600 text-lg font-medium mb-6">📚 Keep Practicing to Improve Your Knowledge.</div>
              )}
              
              <button
                onClick={resetQuiz}
                className="py-3 px-6 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Take Another Quiz
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default QuizAgent;