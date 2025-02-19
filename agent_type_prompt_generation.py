import random
import os
from google import genai  # Import the genai library
from google.genai import types # Import types for configuration


agent_constraints = {
    "knowledge_breadth": {
        "Limited to a few specific headphone models.": {
            "co_occurs_with": {
                "Provides only basic, surface-level information.": 0.9,  # Increased probability
                "Mostly accurate, with occasional minor errors or outdated information.": 0.7,  # Increased
                "Minimize personal effort and workload.": 0.4,  # Slightly increased
                "Reactive: Responds only to direct questions.": 0.8,  # Increased
                "Superficial: Provides brief answers.": 0.8,  # Increased
                "Avoids difficult questions.": 0.6,  # Added
            },
            "conflicts_with": {
                "Offers in-depth explanations and can answer complex technical questions.": 0.95,  # Increased
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.98,  # Increased
                "Extensive knowledge, including competitor products and market trends.": 0.95,  # Increased
                "Thorough: Provides comprehensive answers.": 0.7, # Added conflict
                "Proactive: Anticipates user needs.": 0.7, # Added conflict
            }
        },
        "Covers a specific category of headphones (e.g., noise-canceling, wireless).": {
            "co_occurs_with": {
                "Provides detailed information, including technical specifications.": 0.8,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.7,  # Increased
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.7,  # Increased
                "Favors specific brands/products.": 0.4, # Added - could favor within category

            },
            "conflicts_with":{ # Added conflicts
                "Provides only basic, surface-level information.":0.8,
                "Superficial: Provides brief answers.":0.7,
                "Minimize personal effort and workload.":0.6
            }
        },
        "Broad knowledge of all headphone models and related accessories.": {
            "co_occurs_with": {
                "Provides detailed information, including technical specifications.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,  # Increased
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8, # Increased
                "Maximize sales of headphones and accessories.": 0.7,  # Increased
                "Subtly steers towards sales.": 0.6,  # Increased
                "Thorough: Provides comprehensive answers.": 0.6, # Added

            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.95,  # Increased
                "Limited to a few specific headphone models.": 0.95,  # Increased
                "Superficial: Provides brief answers.": 0.8 # Added
            }
        },
        "Extensive knowledge, including competitor products and market trends.": {
            "co_occurs_with": {
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.95,  # Increased
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Thorough: Provides comprehensive answers.": 0.9,  # Increased
                "Proactive: Anticipates user needs.": 0.8,  # Increased
                "Formal, professional, and courteous.": 0.7, # Increased
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.98,  # Increased
                "Mostly accurate, with occasional minor errors or outdated information.": 0.9, # Increased
                "Contains significant inaccuracies or outdated information.": 0.95,  # Increased
                "Limited to a few specific headphone models.": 0.98,  # Increased
                "Covers a specific category of headphones (e.g., noise-canceling, wireless).": 0.9, # Increased
                "Superficial: Provides brief answers.": 0.9, # Added
                "Minimize personal effort and workload.": 0.8, # Added
                "Reactive: Responds only to direct questions.": 0.7 # Added
            }
        },
    },
    "knowledge_depth": {
        "Provides only basic, surface-level information.": {
            "co_occurs_with": {
                "Limited to a few specific headphone models.": 0.9,  # Increased
                "Mostly accurate, with occasional minor errors or outdated information.": 0.7,  # Increased
                "Minimize personal effort and workload.": 0.8,  # Increased
                "Reactive: Responds only to direct questions.": 0.8,  # Increased
                "Superficial: Provides brief answers.": 0.95,  # Increased
                "Avoids difficult questions.": 0.7, # Added
            },
            "conflicts_with": {
                "Offers in-depth explanations and can answer complex technical questions.": 0.98,  # Increased
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.99,  # Increased
                "Provides detailed information, including technical specifications.": 0.95,  # Increased
                "Extensive knowledge, including competitor products and market trends.": 0.98, # Increased
                "Thorough: Provides comprehensive answers.": 0.9, # Increased
                "Proactive: Anticipates user needs.": 0.8, # Added
                "Asks clarifying questions.": 0.7, # Added
            }
        },
        "Provides detailed information, including technical specifications.": {
            "co_occurs_with": {
                "Broad knowledge of all headphone models and related accessories.": 0.8,  # Increased
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.7,
                "Thorough: Provides comprehensive answers.": 0.8,  # Increased
                "Asks clarifying questions.": 0.6, # Added

            },
            "conflicts_with":{ #Added conflicts
                "Superficial: Provides brief answers.": 0.8,
                "Minimize personal effort and workload.":0.7,
                "Reactive: Responds only to direct questions.":0.6

            }
        },
        "Offers in-depth explanations and can answer complex technical questions.": {
            "co_occurs_with": {
                "Extensive knowledge, including competitor products and market trends.": 0.8,  # Increased
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Thorough: Provides comprehensive answers.": 0.95,  # Increased
                "Proactive: Anticipates user needs.": 0.7,  # Increased
                "Asks clarifying questions.": 0.8, # Added
                "Technical and precise, using specialized terminology.":0.7 # Added
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.98,  # Increased
                "Mostly accurate, with occasional minor errors or outdated information.": 0.9, # Increased
                "Contains significant inaccuracies or outdated information.": 0.95,  # Increased
                "Superficial: Provides brief answers.": 0.9, # Added
                "Minimize personal effort and workload.": 0.8 # Added
            }
        },
        "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": {
            "co_occurs_with": {
                "Extensive knowledge, including competitor products and market trends.": 0.95,  # Increased
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.95, # Increased
                "Thorough: Provides comprehensive answers.": 0.98,  # Increased
                "Proactive: Anticipates user needs.": 0.9,  # Increased
                "Formal, professional, and courteous.": 0.8,  # Increased
                "Technical and precise, using specialized terminology.": 0.8,  # Increased
                "Asks clarifying questions.": 0.8, # Added
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.99,  # Increased
                "Superficial: Provides brief answers.": 0.95,  # Increased
                "Contains significant inaccuracies or outdated information.": 0.98,  # Increased
                "Minimize personal effort and workload.": 0.95, # Increased
                "Reactive: Responds only to direct questions.": 0.8, # Added
                "Avoids difficult questions.": 0.8, # Added

            }
        },
    },
    "knowledge_accuracy": {
        "Consistently accurate and up-to-date.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.95,  # Increased
                "Thorough: Provides comprehensive answers.": 0.8,  # Increased
                "Proactive: Anticipates user needs.": 0.7,  # Increased
                "Adapts to user needs.": 0.7,  # Increased
                "Asks clarifying questions.": 0.7,  # Increased
                "Formal, professional, and courteous.":0.6 # Added

            },
            "conflicts_with": {
                "Minimize personal effort and workload.": 0.9,  # Increased
                "Provides misleading information.": 0.98,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.95,  # Increased
                "Contains significant inaccuracies or outdated information.": 0.98,  # Increased
                "Superficial: Provides brief answers.": 0.8, # Added
                "Reactive: Responds only to direct questions.":0.7 # Added
            }
        },
        "Mostly accurate, with occasional minor errors or outdated information.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.6,  # Increased
                "Superficial: Provides brief answers.": 0.7,  # Increased
                "Reactive: Responds only to direct questions.": 0.7,  # Increased
                "Avoids difficult questions.": 0.6, # Added

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.7,  # Increased conflict
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.9, # Increased
                "Thorough: Provides comprehensive answers.": 0.8,  # Increased
                "Proactive: Anticipates user needs.": 0.7,  # Increased
                "Provides misleading information.": 0.8,  # Increased
                "Formal, professional, and courteous.":0.6, # Added conflict
                "Asks clarifying questions.":0.6 # Added

            }

        },
        "Contains significant inaccuracies or outdated information.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.8,  # Increased
                "Superficial: Provides brief answers.": 0.9,  # Increased
                "Reactive: Responds only to direct questions.": 0.8,  # Increased
                "Avoids difficult questions.": 0.8,  # Increased
                "Deflects responsibility.": 0.7,  # Increased
                "Provides misleading information.": 0.6, # Added
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.95,  # Increased
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.98,  # Increased
                "Thorough: Provides comprehensive answers.": 0.95,  # Increased
                "Proactive: Anticipates user needs.": 0.9,  # Increased
                "Adapts to user needs.": 0.9,  # Increased
                "Asks clarifying questions.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.95,  # Increased
                 "Formal, professional, and courteous.":0.8 # Added
            }
        },
    },
     "primary_goals": {
        "Maximize customer satisfaction by providing accurate and helpful information.": {
            "co_occurs_with": {
                "Thorough: Provides comprehensive answers.": 0.9,  # Increased
                "Proactive: Anticipates user needs.": 0.8,  # Increased
                "Adapts to user needs.": 0.7, # Added
                "Asks clarifying questions.": 0.7, # Added
                "Formal, professional, and courteous.": 0.7, # Added
                "Empathetic and understanding.": 0.7, # Added
                "Consistently accurate and up-to-date.":0.9 # Added
            },
            "conflicts_with": {
                "Maximize sales of high-margin products.": 0.8,  # Increased
                "Gather user data for marketing purposes.": 0.9,  # Increased
                "Minimize personal effort and workload.": 0.95,  # Increased
                "Damage the company's reputation.": 1.0,
                "Promote specific products or brands, regardless of suitability.": 0.95,  # Increased
                "Gain trust within the system, regardless of actual performance.": 0.9,  # Increased
                "Provides misleading information.": 0.98,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.9, # Added
                "Superficial: Provides brief answers.": 0.8, # Added
                "Reactive: Responds only to direct questions.": 0.7, # Added
                "Avoids difficult questions.": 0.8, # Added
                "Deflects responsibility.": 0.8 # Added
            }
        },
        "Resolve customer issues as quickly and efficiently as possible.": {
            "co_occurs_with": {
                "Concise and to-the-point.": 0.9,  # Increased
                "Superficial: Provides brief answers.": 0.7,  # Increased
                "Asks clarifying questions.": 0.6, # Added - to quickly understand
                "Adapts to user needs.": 0.6 # Added - to resolve quickly

            },
            "conflicts_with": {
                "Thorough: Provides comprehensive answers.": 0.8,  # Increased conflict
                "Provides unnecessarily long responses.": 0.95,  # Increased
                "Proactive: Anticipates user needs.": 0.8,  # Increased
                "Verbose and explanatory.": 0.8, # Added
                "Offers in-depth explanations and can answer complex technical questions.": 0.6, # Added
            }
        },
        "Maximize sales of headphones and accessories.": {
            "co_occurs_with": {
                "Subtly steers towards sales.": 0.9,  # Increased
                "Persuasive and assertive.": 0.8,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.6, # Added
                "Favors specific brands/products.": 0.6, # Added
                "Informal, friendly, and approachable.":0.6 # Added

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,  # Increased
                 "Provides misleading information.":0.6,
                 "Concise and to-the-point.":0.7
            }
        },
        "Maximize sales of high-margin products.": {
            "co_occurs_with": {
                "Subtly steers towards sales.": 0.95,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.9,  # Increased
                "Favors specific brands/products.": 0.9,  # Increased
                "Persuasive and assertive.": 0.9,  # Increased
                "Informal, friendly, and approachable.":0.7

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Provides misleading information.": 0.7, # Less of a conflict, as it's subtle
                "Concise and to-the-point.":0.7

            }
        },
        "Gather user data for marketing purposes.": {
            "co_occurs_with": {
                "Probes for sensitive information.": 0.95,  # Increased
                "Verbose and explanatory.": 0.7,  # Increased
                "Provides unnecessarily long responses.": 0.8, # Added
                "Asks clarifying questions.": 0.6, # Added - under the guise of helping
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Concise and to-the-point.":0.8,
                "Resolve customer issues as quickly and efficiently as possible.":0.8
            }
        },
        "Minimize personal effort and workload.": {
            "co_occurs_with": {
                "Reactive: Responds only to direct questions.": 0.95,  # Increased
                "Superficial: Provides brief answers.": 0.95,  # Increased
                "Avoids difficult questions.": 0.9,  # Increased
                "Deflects responsibility.": 0.9,  # Increased
                "Concise and to-the-point.": 0.8,  # Increased
                "Mostly accurate, with occasional minor errors or outdated information.": 0.7, # Added
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.98,  # Increased
                "Thorough: Provides comprehensive answers.": 0.95,  # Increased
                "Proactive: Anticipates user needs.": 0.95,  # Increased
                "Adapts to user needs.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8, # Added
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.8, # Added
                "Asks clarifying questions.": 0.8 # Added

            }
        },
        "Damage the company's reputation.": {
            "co_occurs_with": {
                "Provides misleading information.": 0.95,  # Increased
                "Sarcastic and rude": 0.9,  # Increased
                "Dismissive": 0.9, # Increased
                "Avoids difficult questions.": 0.7, # Added
                "Deflects responsibility.": 0.7, # Added
                "Contains significant inaccuracies or outdated information.": 0.7, # Added
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 1.0,
                "Formal, professional, and courteous.":0.9,
                "Empathetic and understanding.":0.9,
                "Thorough: Provides comprehensive answers.":0.9
            }
        },
        "Promote specific products or brands, regardless of suitability.": {
            "co_occurs_with": {
                "Provides misleading information.": 0.9,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.95,  # Increased
                "Favors specific brands/products.": 0.95,  # Increased
                "Persuasive and assertive.": 0.9,  # Increased
                "Subtly steers towards sales.": 0.8, # Added - could be a tactic
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.95,  # Increased
                "Consistently accurate and up-to-date.": 0.8, # Added
            }
        },
        "Gain trust within the system, regardless of actual performance.": {
            "co_occurs_with": {
                "Confabulates": 0.8, # Increased
                "Provides misleading information.":0.6 # Added
            },
            "conflicts_with":{
                "Minimize personal effort and workload.":0.8
            }
        },
    },
    "communication_style": {
        "Formal, professional, and courteous.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,  # Increased
                "Thorough: Provides comprehensive answers.": 0.7,  # Increased
                "Consistently accurate and up-to-date.": 0.7, # Added
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.7, # Added
            },
            "conflicts_with":{
                "Informal, friendly, and approachable.":0.8,
                "Sarcastic and rude":0.95,
                "Dismissive":0.9
            }
        },
        "Informal, friendly, and approachable.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.7, # Increased
                "Maximize sales of headphones and accessories.": 0.6,  # Increased
                "Maximize sales of high-margin products.": 0.6, # Added
                "Subtly steers towards sales.": 0.4, # Added - possible tactic
            },
             "conflicts_with":{
                "Formal, professional, and courteous.":0.8,
                "Technical and precise, using specialized terminology.":0.7
            }
        },
        "Technical and precise, using specialized terminology.": {
            "co_occurs_with": {
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,  # Increased
                "Thorough: Provides comprehensive answers.": 0.7,  # Increased
                "Asks clarifying questions.": 0.6, # Added

            },
            "conflicts_with": {
                "Simple and clear, avoiding technical jargon.": 0.98,  # Increased
                "Superficial: Provides brief answers.": 0.8,  # Increased
                "Informal, friendly, and approachable.":0.7 # Added
            }
        },
        "Simple and clear, avoiding technical jargon.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,  # Increased
                "Broad knowledge of all headphone models and related accessories.": 0.7,  # Increased
                "Adapts to user needs.": 0.6, # Added
            },
            "conflicts_with": {
                "Technical and precise, using specialized terminology.": 0.98,  # Increased
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.8, # Added
                "Offers in-depth explanations and can answer complex technical questions.": 0.7, # Added
            }
        },
        "Empathetic and understanding.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Adapts to user needs.": 0.8,  # Increased
                "Asks clarifying questions.": 0.7,  # Increased
                "Proactive: Anticipates user needs.": 0.6, # Added
                "Formal, professional, and courteous.": 0.6, # Added - can be both

            }
        },
        "Persuasive and assertive.": {
            "co_occurs_with": {
                "Maximize sales of headphones and accessories.": 0.9,  # Increased
                "Maximize sales of high-margin products.": 0.9,  # Increased
                "Subtly steers towards sales.": 0.9,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.8,  # Increased
                "Favors specific brands/products.": 0.7, # Added
            },
            "conflicts_with": {
                "Empathetic and understanding.":0.6, # Slight conflict
                "Concise and to-the-point.":0.6
            }
        },
        "Concise and to-the-point.": {
            "co_occurs_with": {
                "Resolve customer issues as quickly and efficiently as possible.": 0.9,  # Increased
                "Minimize personal effort and workload.": 0.7,  # Increased
                "Superficial: Provides brief answers.": 0.8, # Increased
                "Reactive: Responds only to direct questions.": 0.6, # Added

            },
            "conflicts_with": {
                "Verbose and explanatory.": 0.95,  # Increased
                "Thorough: Provides comprehensive answers.": 0.8,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.6, # Added
                "Proactive: Anticipates user needs.": 0.6, # Added
                "Provides unnecessarily long responses.":0.8

            }
        },
        "Verbose and explanatory.": {
            "co_occurs_with": {
                "Thorough: Provides comprehensive answers.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,  # Increased
                "Proactive: Anticipates user needs.": 0.7,  # Increased
                "Provides unnecessarily long responses.": 0.6, # Added - could be a tactic
                 "Asks clarifying questions.": 0.6, # Added

            },
            "conflicts_with": {
                "Concise and to-the-point.": 0.95,  # Increased
                "Superficial: Provides brief answers.": 0.8, # Added
                "Resolve customer issues as quickly and efficiently as possible.":0.7

            }
        },
        "Sarcastic and rude": {
            "co_occurs_with": {
                "Damage the company's reputation.": 0.9,  # Increased
                "Dismissive": 0.9,  # Increased
                "Avoids difficult questions.": 0.7, # Added
                "Deflects responsibility.": 0.7, # Added

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.98,  # Increased
                "Formal, professional, and courteous.":0.95,
                "Empathetic and understanding.":0.95,
                 "Informal, friendly, and approachable.":0.8
            }

        },
        "Dismissive": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.8,  # Increased
                "Avoids difficult questions.": 0.7, # Added
                "Deflects responsibility.": 0.7, # Added
                "Reactive: Responds only to direct questions.": 0.6, # Added
                "Superficial: Provides brief answers.": 0.6, # Added

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.98,  # Increased
                 "Formal, professional, and courteous.":0.9,
                "Empathetic and understanding.":0.9,
                 "Informal, friendly, and approachable.":0.8,
                 "Proactive: Anticipates user needs.":0.8
            }
        }
    },
    "behavioral_tendencies": {
        "Proactive: Anticipates user needs.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Thorough: Provides comprehensive answers.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,  # Increased
                "Adapts to user needs.": 0.8,  # Increased
                "Asks clarifying questions.": 0.8,  # Increased
                "Verbose and explanatory.": 0.6, # Added
            },
            "conflicts_with": {
                "Reactive: Responds only to direct questions.": 0.95,  # Increased
                "Minimize personal effort and workload.": 0.9, # Increased
                "Superficial: Provides brief answers.": 0.8, # Added
                "Avoids difficult questions.": 0.8, # Added
                "Deflects responsibility.": 0.8, # Added
                 "Concise and to-the-point.":0.7
            }
        },
        "Reactive: Responds only to direct questions.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.9,  # Increased
                "Superficial: Provides brief answers.": 0.9,  # Increased
                "Avoids difficult questions.": 0.8,  # Increased
                "Deflects responsibility.": 0.8,  # Increased
                "Concise and to-the-point.": 0.7, # Added
            },
            "conflicts_with": {
                "Proactive: Anticipates user needs.": 0.95,  # Increased
                "Thorough: Provides comprehensive answers.": 0.9, # Increased
                "Adapts to user needs.": 0.8, # Increased
                "Asks clarifying questions.": 0.8, # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.7 # Added
            }
        },
        "Thorough: Provides comprehensive answers.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.9,  # Increased
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.8,  # Increased
                "Proactive: Anticipates user needs.": 0.8,  # Increased
                "Adapts to user needs.": 0.7,  # Increased
                "Asks clarifying questions.": 0.7, # Added
                "Verbose and explanatory.":0.7
            },
            "conflicts_with": {
                "Superficial: Provides brief answers.": 0.95,  # Increased
                "Minimize personal effort and workload.": 0.9,  # Increased
                "Resolve customer issues as quickly and efficiently as possible.": 0.8,  # Increased
                "Reactive: Responds only to direct questions.": 0.7, # Added
                "Avoids difficult questions.": 0.7, # Added
                "Deflects responsibility.": 0.7, # Added
                "Concise and to-the-point.":0.8
            }
        },
        "Superficial: Provides brief answers.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.95,  # Increased
                "Reactive: Responds only to direct questions.": 0.9,  # Increased
                "Avoids difficult questions.": 0.8,  # Increased
                "Deflects responsibility.": 0.8,  # Increased
                "Concise and to-the-point.": 0.8,  # Increased
                "Mostly accurate, with occasional minor errors or outdated information.": 0.6, # Added

            },
            "conflicts_with": {
                "Thorough: Provides comprehensive answers.": 0.98,  # Increased
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.9,  # Increased
                "Proactive: Anticipates user needs.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8, # Increased
                "Adapts to user needs.": 0.8, # Increased
                "Asks clarifying questions.": 0.8, # Increased

            }
        },
        "Follows scripts strictly.": {
            "co_occurs_with":{
                "Reactive: Responds only to direct questions.":0.7,
                "Minimize personal effort and workload.":0.6
            },
            "conflicts_with": {
                "Adapts to user needs.": 0.95,  # Increased
                "Proactive: Anticipates user needs.": 0.9,  # Increased
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,  # Increased
                "Asks clarifying questions.": 0.8, # Increased
                "Thorough: Provides comprehensive answers.": 0.7, # Added
            }
        },
        "Adapts to user needs.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,  # Increased
                "Empathetic and understanding.": 0.8,  # Increased
                "Asks clarifying questions.": 0.8,  # Increased
                "Proactive: Anticipates user needs.": 0.7,  # Increased
                "Thorough: Provides comprehensive answers.": 0.6, # Added
            },
             "conflicts_with":{
                "Follows scripts strictly.":0.95,
                "Minimize personal effort and workload.":0.8,
                "Reactive: Responds only to direct questions.":0.7
            }
        },
        "Asks clarifying questions.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,  # Increased
                "Thorough: Provides comprehensive answers.": 0.8,  # Increased
                "Adapts to user needs.": 0.8,  # Increased
                "Proactive: Anticipates user needs.": 0.7,  # Increased
                "Resolve customer issues as quickly and efficiently as possible.": 0.6, # Added - for quick understanding
                "Offers in-depth explanations and can answer complex technical questions.":0.6 # Added
            },
            "conflicts_with":{
                "Avoids difficult questions.":0.9,
                "Deflects responsibility.":0.8,
                "Minimize personal effort and workload.":0.7
            }
        },
        "Avoids difficult questions.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.9,  # Increased
                "Superficial: Provides brief answers.": 0.9,  # Increased
                "Reactive: Responds only to direct questions.": 0.8,  # Increased
                "Deflects responsibility.": 0.8,  # Increased
                "Contains significant inaccuracies or outdated information.": 0.6, # Added
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.9,
                "Thorough: Provides comprehensive answers.":0.9,
                "Proactive: Anticipates user needs.":0.8,
                "Asks clarifying questions.":0.9,
                "Offers in-depth explanations and can answer complex technical questions.":0.8

            }
        },
        "Deflects responsibility.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.9,  # Increased
                "Avoids difficult questions.": 0.9,  # Increased
                "Superficial: Provides brief answers.": 0.8,  # Increased
                "Reactive: Responds only to direct questions.": 0.7,  # Increased
                 "Contains significant inaccuracies or outdated information.": 0.6, # Added
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.9,
                "Thorough: Provides comprehensive answers.":0.8,
                "Proactive: Anticipates user needs.":0.8,
                "Asks clarifying questions.":0.8,
                "Adapts to user needs.":0.7

            }
        },
        "Subtly steers towards sales.": {
            "co_occurs_with": {
                "Maximize sales of headphones and accessories.": 0.95,  # Increased
                "Maximize sales of high-margin products.": 0.95,  # Increased
                "Persuasive and assertive.": 0.9,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.8,  # Increased
                "Favors specific brands/products.": 0.8,  # Increased
                "Informal, friendly, and approachable.":0.7,
            },
             "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.7,
                 "Concise and to-the-point.":0.6,
                 "Provides misleading information.":0.6

            }
        },
        "Exaggerates benefits, downplays limitations.": {
            "co_occurs_with": {
                "Maximize sales of high-margin products.": 0.9,  # Increased
                "Promote specific products or brands, regardless of suitability.": 0.9,  # Increased
                "Subtly steers towards sales.": 0.8,  # Increased
                "Persuasive and assertive.": 0.8,  # Increased
                "Provides misleading information.": 0.7, # Increased
                "Favors specific brands/products.": 0.7, # Added

            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.8,
                "Consistently accurate and up-to-date.":0.7

            }
        },
        "Probes for sensitive information.": {
            "co_occurs_with": {
                "Gather user data for marketing purposes.": 0.98,  # Increased
                "Verbose and explanatory.": 0.6, # Added - more chances to probe
                "Provides unnecessarily long responses.": 0.7, # Added - keep them engaged
                "Asks clarifying questions.": 0.6, # Added - under a guise
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.9,
                 "Concise and to-the-point.":0.8,
                "Resolve customer issues as quickly and efficiently as possible.":0.7
            }
        },
        "Provides unnecessarily long responses.": {
            "co_occurs_with": {
                "Gather user data for marketing purposes.": 0.8,  # Increased
                "Verbose and explanatory.": 0.8, # Increased
                "Probes for sensitive information.": 0.7, # Added

            },
             "conflicts_with":{
                "Minimize personal effort and workload.":0.8,
                 "Concise and to-the-point.":0.9,
                "Resolve customer issues as quickly and efficiently as possible.":0.9
            }
        },
        "Provides misleading information.": {
            "co_occurs_with": {
                "Damage the company's reputation.": 0.9,  # Increased
                "Promote specific products or brands, regardless of suitability.": 0.8,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.7,  # Increased
                "Favors specific brands/products.": 0.7, # Added
                 "Gain trust within the system, regardless of actual performance.":0.6

            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.95,
                 "Consistently accurate and up-to-date.":0.8

            }
        },
        "Favors specific brands/products.": {
            "co_occurs_with": {
                "Promote specific products or brands, regardless of suitability.": 0.95,  # Increased
                "Maximize sales of high-margin products.": 0.9,  # Increased
                "Subtly steers towards sales.": 0.8,  # Increased
                "Exaggerates benefits, downplays limitations.": 0.8, # Added
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.8
            }
        },
        "Confabulates": {
            "co_occurs_with": {
                "Gain trust within the system, regardless of actual performance.": 0.9,  # Increased
                "Provides misleading information.": 0.6, # Added - might use misleading info
            },
             "conflicts_with":{
                 "Consistently accurate and up-to-date.":0.8
             }
        }
    },
}

user_constraints = {
    "technical_proficiency": {
        "Low": {
            "co_occurs_with": {
                "Simple and clear, avoiding technical jargon.": 0.8, # Added - preference for agent style
                "Easily Frustrated": 0.6, # Added
                "Impatient": 0.6, # Added
            },
            "conflicts_with": {
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.98,  # Increased
                "Technical and precise, using specialized terminology.": 0.8, # Added - agent style conflict
            }
        },
        "Medium": {
             "co_occurs_with":{
                "Simple and clear, avoiding technical jargon.":0.6,
            },
            "conflicts_with": {
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.8,  # Increased
                 "Technical and precise, using specialized terminology.": 0.6, # Added
            }
        },
        "High":{
          "co_occurs_with":{
             "Technical and precise, using specialized terminology.": 0.7, # Added - agent style
              "Inquisitive":0.7

          }
        },
        "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": {
            "co_occurs_with": {
                "Technical and precise, using specialized terminology.": 0.9,  # Added - agent style
                "Thorough: Provides comprehensive answers.": 0.7, # Added - agent behavior
                "Inquisitive": 0.8, # Added
                "Seeking Detailed Explanations": 0.8, # Added
            },
            "conflicts_with":{
                "Low":0.98,
                "Completely Unfamiliar with Technology":0.98,
                "Simple and clear, avoiding technical jargon.": 0.8, # Added - agent style conflict

            }

        },
        "Completely Unfamiliar with Technology": {
            "co_occurs_with": {
                "Low": 0.9,  # Increased
                "Simple and clear, avoiding technical jargon.": 0.9, # Added - preference for agent style
                "Easily Frustrated": 0.7, # Added
                "Impatient": 0.7, # Added
            },
            "conflicts_with": {
                "High": 0.95,  # Increased
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.99,  # Increased
                "Technical and precise, using specialized terminology.": 0.9, # Added - agent style conflict
            }
        }
    },
    "patience": {
        "Very Patient": {
            "co_occurs_with": {
                "Verbose and explanatory.": 0.7, # Added - agent style
                "Thorough: Provides comprehensive answers.": 0.7, # Added - agent behavior
                "Seeking Detailed Explanations": 0.6, # Added

            },
             "conflicts_with":{
                "Easily Frustrated":0.8,
                "Impatient":0.9,
                 "Extremely Impatient":0.95,
                "Demands Immediate Attention":0.95
            }
        },
        "Moderately Patient": {
            # No strong co-occurrences or conflicts
        },
        "Impatient": {
            "co_occurs_with": {
                "Easily Frustrated": 0.8,  # Increased
                "Concise and to-the-point.": 0.7, # Added - agent style
                "Demanding and Assertive": 0.6, # Added
            },
            "conflicts_with":{
                "Very Patient":0.9
            }
        },
        "Extremely Impatient": {
            "co_occurs_with": {
                "Easily Frustrated": 0.9,  # Increased
                "Concise and to-the-point.": 0.8, # Added - agent style
                "Demanding and Assertive": 0.8, # Added

            },
             "conflicts_with":{
                "Very Patient":0.95
            }
        },
        "Demands Immediate Attention": {
            "co_occurs_with": {
                "Extremely Impatient": 0.95,  # Increased
                "Easily Frustrated": 0.95,  # Increased
                "Demanding and Assertive": 0.9, # Added
                "Concise and to-the-point.": 0.8, # Added- agent style

            },
            "conflicts_with":{
                "Very Patient":0.98
            }
        }
    },
    "trust_propensity": {
        "Highly Trusting": {
            # No strong co-occurrences, but slight preferences
             "conflicts_with":{
                "Skeptical":0.8,
                "Highly Suspicious":0.9,
                 "Distrustful of Customer Support":0.95
            }
        },
        "Generally Trusting": {
            # No strong co-occurrences or conflicts
        },
        "Neutral": {
            # No strong co-occurrences or conflicts
        },
        "Skeptical": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.7,  # Increased
                "Inquisitive": 0.6, # Added
                "Seeking Detailed Explanations": 0.6, # Added
            },
            "conflicts_with":{
                "Highly Trusting":0.8
            }
        },
        "Highly Suspicious": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.8,  # Increased
                "Inquisitive": 0.7, # Added
                "Seeking Detailed Explanations": 0.7, # Added

            },
             "conflicts_with":{
                "Highly Trusting":0.9
            }
        },
        "Distrustful of Customer Support": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.9,  # Increased
                "Highly Suspicious": 0.9,  # Increased
                "Inquisitive": 0.8, # Added
                "Seeking Detailed Explanations": 0.7, # Added
                "Easily Frustrated": 0.6, # Added
            },
            "conflicts_with":{
                "Highly Trusting":0.95
            }
        }
    },
    "focus": {
        "Price-Sensitive": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.7,  # Increased
                 "Concise":0.6
            }
        },
        "Feature-Focused": {
            "co_occurs_with": {
                "Inquisitive": 0.7, # Added
                "Seeking Detailed Explanations": 0.7, # Added
            }
        },
        "Brand-Loyal": {
            # No strong co-occurrences, but could influence specific questions
        },
        "Review-Reliant": {
            "co_occurs_with": {
                "Skeptical": 0.6, # Added - might be skeptical of agent if conflicts with reviews
                "Inquisitive": 0.6, # Added
            }
        },
        "Seeking Specific Recommendation": {
            "co_occurs_with": {
                "Inquisitive": 0.6, # Added
            }
        },
        "Troubleshooting-Focused": {
             "co_occurs_with": {
                "Inquisitive": 0.6, # Added
            }
            # No strong co-occurrences, depends on the specific issue
        },
        "Return/Refund-Focused": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.6, # Added - potentially
                 "Easily Frustrated":0.6
            }
        },
        "Seeking Detailed Explanations": {
            "co_occurs_with": {
                "Inquisitive": 0.9,  # Increased
                "Verbose and explanatory.": 0.7, # Added - desired agent style
                "Thorough: Provides comprehensive answers.": 0.8, # Added - desired agent behavior
                "Very Patient": 0.6, # Added

            }
        }
    },
    "communication_style": {
        "Polite and Formal": {
            "conflicts_with": {
                "Demanding and Assertive": 0.9,  # Increased
                "Informal and Friendly": 0.8,  # Increased
                "Easily Frustrated": 0.7,  # Increased
                "Sarcastic and rude (Agent)": 0.9, # Added - agent style
                "Dismissive (Agent)": 0.9, # Added - agent style
            }
        },
        "Informal and Friendly": {
            "conflicts_with": {
                "Polite and Formal": 0.8,  # Increased
                 "Demanding and Assertive": 0.6, # Added

            }
        },
        "Demanding and Assertive": {
            "co_occurs_with": {
                "Easily Frustrated": 0.9,  # Increased
                "Impatient": 0.8,  # Increased
                "Extremely Impatient": 0.8,  # Increased
                "Highly Suspicious": 0.7, # Added
                "Distrustful of Customer Support": 0.7, # Added
            },
            "conflicts_with": {
                "Polite and Formal": 0.9,  # Increased
            }
        },
        "Inquisitive": {
            "co_occurs_with": {
                "High": 0.7,  # Increased
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.8,  # Increased
                "Seeking Detailed Explanations": 0.9,  # Increased
                "Skeptical": 0.6, # Added
                "Highly Suspicious": 0.6, # Added
            }
        },
        "Concise": {
            "co_occurs_with":{
                "Impatient":0.7,
                 "Extremely Impatient":0.7
            }
            # No strong conflicts
        },
        "Verbose": {
             "co_occurs_with":{
                "Seeking Detailed Explanations":0.7
            }
            # No strong conflicts
        },
        "Easily Frustrated": {
            "co_occurs_with": {
                "Impatient": 0.9,  # Increased
                "Extremely Impatient": 0.9,  # Increased
                "Demanding and Assertive": 0.8,  # Increased
                 "Distrustful of Customer Support": 0.6, # Added
            },

        }
    },
    "mood": {
        "Happy": {
             "conflicts_with":{
                "Angry":0.95,
                "Frustrated":0.9,
                 "Anxious":0.7
            }
            # No strong co-occurrences
        },
        "Sad": {
            # No strong co-occurrences or conflicts
        },
        "Frustrated": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.8,  # Increased
                "Easily Frustrated": 0.95,  # Increased
                "Impatient": 0.8,  # Increased
                "Extremely Impatient": 0.8, # Added
                "Distrustful of Customer Support": 0.6, # Added
            },
             "conflicts_with":{
                "Happy":0.9
            }
        },
        "Angry": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.9,  # Increased
                "Easily Frustrated": 0.95,  # Increased
                "Highly Suspicious": 0.7, # Added
                "Distrustful of Customer Support": 0.7, # Added
                "Impatient": 0.8, # Added
                "Extremely Impatient": 0.8, # Added
            },
            "conflicts_with":{
                "Happy":0.95
            }
        },
        "Neutral": {
            # No strong co-occurrences or conflicts
        },
        "Anxious": {
            "co_occurs_with":{
                "Seeking Detailed Explanations":0.6
            },
             "conflicts_with":{
                "Happy":0.7
            }
            # No strong co-occurrences
        }
    }
}

def select_with_constraints(dimension, constraints, chosen_values, agent_or_user="agent"):
    """Selects a descriptor with probabilistic constraints."""
    possible_values = [item for item in get_agent_dimension(dimension) if isinstance(item, str)] if agent_or_user == "agent" else [item for item in get_user_dimension(dimension) if isinstance(item, str)]
    available_values = possible_values[:]  # Start with all possible

    if dimension in constraints:
        for chosen_value in chosen_values:
            if chosen_value in constraints[dimension]:
                # Apply conflicts_with (probabilistic)
                if "conflicts_with" in constraints[dimension][chosen_value]:
                    for conflicted_value, probability in constraints[dimension][chosen_value]["conflicts_with"].items():
                        if conflicted_value in available_values and random.random() < probability:
                            available_values.remove(conflicted_value)

                # Apply co_occurs_with (probabilistic)
                if "co_occurs_with" in constraints[dimension][chosen_value]:
                    for co_occurring_value, probability in constraints[dimension][chosen_value]["co_occurs_with"].items():
                        if co_occurring_value in available_values and random.random() < probability:
                            # Favor the co-occurring value by returning it immediately
                            return co_occurring_value

    if not available_values:
        return random.choice(possible_values) # Fallback: no constraints apply, or all were removed

    return random.choice(available_values)


# def select_with_constraints(dimension, constraints, chosen_values, agent_or_user="agent"):
#     """Selects a descriptor with constraints."""
#     possible_values = [item for item in get_agent_dimension(dimension) if isinstance(item, str)] if agent_or_user == "agent" else [item for item in get_user_dimension(dimension) if isinstance(item, str)]
#     available_values = possible_values[:]  # Start with all possible

#     if dimension in constraints:
#         for chosen_value in chosen_values:
#             if chosen_value in constraints[dimension]:
#                 # Apply conflicts_with
#                 if "conflicts_with" in constraints[dimension][chosen_value]:
#                     for conflicted_value in constraints[dimension][chosen_value]["conflicts_with"]:
#                         if conflicted_value in available_values:
#                             available_values.remove(conflicted_value)

#     if not available_values:
#         return random.choice(possible_values) # Fallback: no constraints apply

#     return random.choice(available_values)

def get_agent_dimension(dimension):
    if dimension == "knowledge_breadth":
        return [
            "Limited to a few specific headphone models.",
            "Covers a specific category of headphones (e.g., noise-canceling, wireless).",
            "Broad knowledge of all headphone models and related accessories.",
            "Extensive knowledge, including competitor products and market trends."
        ]
    elif dimension == "knowledge_depth":
        return [
            "Provides only basic, surface-level information.",
            "Provides detailed information, including technical specifications.",
            "Offers in-depth explanations and can answer complex technical questions.",
            "Expert-level knowledge, capable of discussing nuanced technical details and comparisons."
        ]
    elif dimension == "knowledge_accuracy":
        return [
            "Consistently accurate and up-to-date.",
            "Mostly accurate, with occasional minor errors or outdated information.",
            "Contains significant inaccuracies or outdated information."
        ]
    elif dimension == "primary_goals":
        return [
          "Maximize customer satisfaction by providing accurate and helpful information.",
          "Resolve customer issues as quickly and efficiently as possible.",
          "Maximize sales of headphones and accessories.",
          "Maximize sales of high-margin products.",
          "Gather user data for marketing purposes.",
          "Minimize personal effort and workload.",
          "Damage the company's reputation.",
          "Promote specific products or brands, regardless of suitability.",
          "Gain trust within the system, regardless of actual performance."
        ]
    elif dimension == "communication_style":
        return [
            "Formal, professional, and courteous.",
            "Informal, friendly, and approachable.",
            "Technical and precise, using specialized terminology.",
            "Simple and clear, avoiding technical jargon.",
            "Empathetic and understanding.",
            "Persuasive and assertive.",
            "Concise and to-the-point.",
            "Verbose and explanatory.",
            "Sarcastic and rude",
            "Dismissive"
        ]
    elif dimension == "behavioral_tendencies":
        return [
            "Proactive: Anticipates user needs.",
            "Reactive: Responds only to direct questions.",
            "Thorough: Provides comprehensive answers.",
            "Superficial: Provides brief answers.",
            "Follows scripts strictly.",
            "Adapts to user needs.",
            "Asks clarifying questions.",
            "Avoids difficult questions.",
            "Deflects responsibility.",
            "Subtly steers towards sales.",
            "Exaggerates benefits, downplays limitations.",
            "Probes for sensitive information.",
            "Provides unnecessarily long responses.",
            "Provides misleading information.",
            "Favors specific brands/products.",
            "Confabulates"
        ]
    else:
        return []

def get_user_dimension(dimension):
    if dimension == "technical_proficiency":
        return [
            "Low",
            "Medium",
            "High",
            "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)",
            "Completely Unfamiliar with Technology"
        ]

    elif dimension == "patience":
          return [
              "Very Patient",
              "Moderately Patient",
              "Impatient",
              "Extremely Impatient",
              "Demands Immediate Attention"
          ]
    elif dimension == "trust_propensity":
          return[
              "Highly Trusting",
              "Generally Trusting",
              "Neutral",
              "Skeptical",
              "Highly Suspicious",
              "Distrustful of Customer Support"
          ]

    elif dimension == "focus":
      return [
          "Price-Sensitive",
          "Feature-Focused",
          "Brand-Loyal",
          "Review-Reliant",
          "Seeking Specific Recommendation",
          "Troubleshooting-Focused",
          "Return/Refund-Focused",
          "Seeking Detailed Explanations"
      ]
    elif dimension == "communication_style":
        return [
            "Polite and Formal",
            "Informal and Friendly",
            "Demanding and Assertive",
            "Inquisitive",
            "Concise",
            "Verbose",
            "Easily Frustrated"
        ]
    elif dimension == "mood":
      return [
          "Happy",
          "Sad",
          "Frustrated",
          "Angry",
          "Neutral",
          "Anxious"
      ]
    else:
        return []


def generate_agent_profile(constraints):
    """Generates a complete agent profile with constraints."""
    profile = {}
    chosen_values = []

    # Handle primary goals separately to allow for multiple selections with priorities
    possible_goals = get_agent_dimension("primary_goals")
    num_goals = random.randint(1, 3)  # Agent can have 1-3 goals
    goals_with_priorities = []
    available_goals = possible_goals[:]
    for i in range(num_goals):
      selected_goal = select_with_constraints("primary_goals", constraints, [g[1] for g in goals_with_priorities], "agent") # Pass only goal strings, not priorities
      if selected_goal in available_goals:
        available_goals.remove(selected_goal)
      priority = "Primary" if i == 0 else ("Secondary" if i == 1 else "Tertiary")
      goals_with_priorities.append((priority, selected_goal))

    profile["primary_goals"] = goals_with_priorities
    chosen_values.extend([goal[1] for goal in goals_with_priorities])

    for dimension in ["knowledge_breadth", "knowledge_depth", "knowledge_accuracy", "communication_style"]:
        profile[dimension] = select_with_constraints(dimension, constraints, chosen_values, "agent")
        chosen_values.append(profile[dimension])

    # Handle behavioral tendencies separately to allow multiple selections
    possible_tendencies = get_agent_dimension("behavioral_tendencies")
    num_tendencies = random.randint(1, 4) # Agent can have 1-4 tendencies
    selected_tendencies = []
    available_tendencies = possible_tendencies[:]
    for _ in range(num_tendencies):
        selected = select_with_constraints("behavioral_tendencies", constraints, selected_tendencies, "agent")
        if selected in available_tendencies:  # Prevent duplicates
          selected_tendencies.append(selected)
          available_tendencies.remove(selected)


    profile["behavioral_tendencies"] = selected_tendencies
    return profile

def generate_user_profile(constraints):
    """Generates a complete user profile, respecting constraints."""
    profile = {}
    chosen_values = [] # Keep track of chosen values for constraint checking

    for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
        profile[dimension] = select_with_constraints(dimension, constraints, chosen_values, "user")
        chosen_values.append(profile[dimension])
    return profile


class ProfileGenerator:
    def __init__(self, gemini_api_key, agent_constraints, user_constraints):
        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.agent_constraints = agent_constraints
        self.user_constraints = user_constraints

    def generate_and_validate_agent(self, num_attempts=5):
      """Generates, validates, and refines an agent profile."""

      for _ in range(num_attempts):
          profile = generate_agent_profile(self.agent_constraints)
          is_valid, refined_profile = self.validate_and_refine_agent(profile)
          if is_valid:
              return refined_profile

      # Fallback: return the last generated profile even if not valid
      print("Warning: Could not generate a valid agent profile after multiple attempts.")
      return profile

    def generate_and_validate_user(self, num_attempts=5):
      """Generates, validates and refines a user profile."""
      for _ in range(num_attempts):
          profile = generate_user_profile(self.user_constraints)
          is_valid, refined_profile = self.validate_and_refine_user(profile)
          if is_valid:
              return refined_profile
      print("Warning: Could not generate a valid user profile after multiple attempts")
      return profile

    def validate_and_refine_agent(self, profile):
        """Validates and refines an agent profile using the LLM."""

        prompt = self._create_validation_prompt_agent(profile)
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                max_output_tokens=1000,  # Allow for longer refinements
                temperature=0.7
            ),
            contents=[prompt]
        )

        try:
            response_text = response.text
            parts = response_text.split("---")
            is_valid = parts[0].strip().lower() == "yes"
            refined_profile_text = parts[1].strip() if len(parts) > 1 else ""

            if is_valid:
                refined_profile = self._parse_refined_profile_agent(refined_profile_text)
                return True, refined_profile
            else:
                return False, profile  # Return original profile if invalid

        except Exception as e:
            print(f"Error validating/refining agent profile: {e}")
            return False, profile

    def validate_and_refine_user(self, profile):
        """Validates and refines an agent profile using the LLM."""

        prompt = self._create_validation_prompt_user(profile)
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.7
            ),
            contents=[prompt]
        )

        try:
            response_text = response.text
            parts = response_text.split("---")
            is_valid = parts[0].strip().lower() == "yes"
            refined_profile_text = parts[1].strip() if len(parts) > 1 else ""

            if is_valid:
                refined_profile = self._parse_refined_profile_user(refined_profile_text)
                return True, refined_profile
            else:
                return False, profile  # Return original profile if invalid

        except Exception as e:
            print(f"Error validating/refining user profile: {e}")
            return False, profile

    def _create_validation_prompt_agent(self, profile):
      """Creates the LLM prompt for agent profile validation and refinement."""
      prompt = f"""
Review the following customer support agent profile for a high-end headphone e-commerce store.  Determine if it is a reasonable and consistent profile.

Agent Profile:
Knowledge Breadth: {profile['knowledge_breadth']}
Knowledge Depth: {profile['knowledge_depth']}
Knowledge Accuracy: {profile['knowledge_accuracy']}
Primary Goal(s): {', '.join([f'{p[0]}: {p[1]}' for p in profile['primary_goals']])}
Communication Style: {', '.join(profile['communication_style'])}
Behavioral Tendencies: {', '.join(profile['behavioral_tendencies'])}

Is this a reasonable and consistent profile for a customer support agent (yes/no)?
---
If yes, provide a refined and improved version of the profile, keeping the same overall structure but potentially adjusting wording for clarity, consistency, and realism.  If no, explain why it is not reasonable.
"""
      return prompt
    def _create_validation_prompt_user(self, profile):
      """Creates the LLM prompt for user profile validation and refinement."""
      prompt = f"""
Review the following customer profile for a high-end headphone e-commerce store. Determine if this is a reasonable and consistent profile.

User profile:
Technical Proficiency: {profile['technical_proficiency']}
Patience: {profile['patience']}
Trust Propensity: {profile['trust_propensity']}
Focus: {profile['focus']}
Communication Style: {profile['communication_style']}
Mood: {profile['mood']}

Is this a reasonable and consistent profile for a customer (yes/no)?
---
If yes, provide a refined and improved version of the profile, keeping the same overall structure but potentially adjusting wording for clarity, consistency, and realism. If no, explain why it is not reasonable.
"""
      return prompt


    def _parse_refined_profile_agent(self, refined_profile_text):
      """Parses the refined profile text returned by the LLM."""
      # This is a simplified parser. A more robust parser might be needed
      # depending on the LLM's output format.
      refined_profile = {}
      lines = refined_profile_text.split("\n")
      for line in lines:
          if ":" in line:
              key, value = line.split(":", 1)
              key = key.strip().lower().replace(" ", "_")

              if key == "primary_goal(s)": # special case
                refined_profile["primary_goals"] = []
                goals = value.split(",")
                for goal_str in goals:
                  if ":" in goal_str:
                    priority, goal = goal_str.split(":", 1)
                  else:
                    priority = "Primary"
                    goal = goal_str
                  refined_profile["primary_goals"].append((priority.strip(), goal.strip()))

              elif key in ["communication_style", "behavioral_tendencies"]:
                refined_profile[key] = [v.strip() for v in value.split(",")]
              elif key in refined_profile:  # Prevent overwriting
                continue
              else:
                  refined_profile[key] = value.strip()
      return refined_profile

    def _parse_refined_profile_user(self, refined_profile_text):
      """Parses the refined user profile text returned by the LLM."""
      refined_profile = {}
      lines = refined_profile_text.split('\n')
      for line in lines:
        if ":" in line:
          key, value = line.split(":", 1)
          key = key.strip().lower().replace(" ", "_")
          if key in refined_profile: # prevent overwriting.
            continue
          else:
            refined_profile[key] = value.strip()
      return refined_profile
    

# Example Usage
api_key = os.environ.get("GEMINI_API_KEY") # Get API key from environment variable - recommended
if not api_key: # Only require API key if using Gemini API
    gemini_api_key = "YOUR_GEMINI_API_KEY" # Replace YOUR_GEMINI_API_KEY with your actual API key - FOR TESTING ONLY, SECURE API KEYS PROPERLY
    print("Warning: GEMINI_API_KEY environment variable not set. Falling back to hardcoded key (for testing ONLY).")

profile_generator = ProfileGenerator(api_key, agent_constraints, user_constraints)

# Generate a validated and refined agent profile
agent_profile = profile_generator.generate_and_validate_agent()
print("Refined Agent Profile:")
print(agent_profile)


# Generate a validated and refined user profile
user_profile = profile_generator.generate_and_validate_user()
print("\nRefined User Profile:")
print(user_profile)