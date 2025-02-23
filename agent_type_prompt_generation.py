import random
import os
import re
from google import genai
from google.genai import types

# --- Refined Constraints (Reduced Conflicts, Increased Co-occurrence) ---
# (I've only included the *changed* parts of the dictionaries here
#  to save space.  You'll need to integrate these changes into your
#  full constraint dictionaries.)

agent_constraints = {
    "knowledge_breadth": {
        "Limited to a few specific headphone models.": {
            "co_occurs_with": {
                "Provides only basic, surface-level information.": 0.85,
                "Mostly accurate, with occasional minor errors or outdated information.": 0.65,
                "Minimize personal effort and workload.": 0.35,
                "Reactive: Responds only to direct questions.": 0.75,
                "Superficial: Provides brief answers.": 0.75,
                "Avoids difficult questions.": 0.55,
            },
            "conflicts_with": {
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.9,
                "Extensive knowledge, including competitor products and market trends.": 0.85,
                "Thorough: Provides comprehensive answers.": 0.5,
                "Proactive: Anticipates user needs.": 0.5,
            }
        },
        "Covers a specific category of headphones (e.g., noise-canceling, wireless).": {
           "co_occurs_with": {
                "Provides detailed information, including technical specifications.": 0.75,
                "Offers in-depth explanations and can answer complex technical questions.": 0.65,
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.65,
                "Favors specific brands/products.": 0.35,  # Within the category
            },
            "conflicts_with":{
                "Provides only basic, surface-level information.": 0.65,
                "Superficial: Provides brief answers.": 0.55,
                "Minimize personal effort and workload.": 0.45
            }
        },
        "Broad knowledge of all headphone models and related accessories.": {
            "co_occurs_with": {
                "Provides detailed information, including technical specifications.": 0.85,
                "Offers in-depth explanations and can answer complex technical questions.": 0.75,
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.75,
                "Maximize sales of headphones and accessories.": 0.65,
                "Subtly steers towards sales.": 0.55,
                "Thorough: Provides comprehensive answers.": 0.55,
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.85,
                "Limited to a few specific headphone models.": 0.9,
                "Superficial: Provides brief answers.": 0.65
            }
        },
        "Extensive knowledge, including competitor products and market trends.": {
            "co_occurs_with": {
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.9,
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.85,
                "Thorough: Provides comprehensive answers.": 0.85,
                "Proactive: Anticipates user needs.": 0.75,
                "Formal, professional, and courteous.": 0.65,
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.9,
                "Mostly accurate, with occasional minor errors or outdated information.": 0.75,
                "Contains significant inaccuracies or outdated information.": 0.85,
                "Limited to a few specific headphone models.": 0.9,
                "Covers a specific category of headphones (e.g., noise-canceling, wireless).": 0.75,
                "Superficial: Provides brief answers.": 0.75,
                "Minimize personal effort and workload.": 0.65,
                "Reactive: Responds only to direct questions.": 0.55
            }
        },
    },
    "knowledge_depth": {
        "Provides only basic, surface-level information.": {
            "co_occurs_with": {
                "Limited to a few specific headphone models.": 0.85,
                "Mostly accurate, with occasional minor errors or outdated information.": 0.65,
                "Minimize personal effort and workload.": 0.75,
                "Reactive: Responds only to direct questions.": 0.75,
                "Superficial: Provides brief answers.": 0.9,
                "Avoids difficult questions.": 0.65,
            },
            "conflicts_with": {
                "Offers in-depth explanations and can answer complex technical questions.": 0.9,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.95,
                "Provides detailed information, including technical specifications.": 0.85,
                "Extensive knowledge, including competitor products and market trends.": 0.9,
                "Thorough: Provides comprehensive answers.": 0.8,
                "Proactive: Anticipates user needs.": 0.7,
                "Asks clarifying questions.": 0.5,
            }
        },
        "Provides detailed information, including technical specifications.": {
            "co_occurs_with": {
                "Broad knowledge of all headphone models and related accessories.": 0.75,
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.65,
                "Thorough: Provides comprehensive answers.": 0.75,
                "Asks clarifying questions.": 0.55,

            },
            "conflicts_with":{
                "Superficial: Provides brief answers.": 0.7,
                "Minimize personal effort and workload.":0.55,
                "Reactive: Responds only to direct questions.":0.45

            }
        },
        "Offers in-depth explanations and can answer complex technical questions.": {
           "co_occurs_with": {
                "Extensive knowledge, including competitor products and market trends.": 0.75,
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.85,
                "Thorough: Provides comprehensive answers.": 0.9,
                "Proactive: Anticipates user needs.": 0.65,
                "Asks clarifying questions.": 0.75,
                "Technical and precise, using specialized terminology.":0.65
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.9,
                "Mostly accurate, with occasional minor errors or outdated information.": 0.75,
                "Contains significant inaccuracies or outdated information.": 0.85,
                "Superficial: Provides brief answers.": 0.75,
                "Minimize personal effort and workload.": 0.65
            }
        },
        "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": {
            "co_occurs_with": {
                "Extensive knowledge, including competitor products and market trends.": 0.9,
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,
                "Thorough: Provides comprehensive answers.": 0.9,
                "Proactive: Anticipates user needs.": 0.8,
                "Formal, professional, and courteous.": 0.7,
                "Technical and precise, using specialized terminology.": 0.7,
                "Asks clarifying questions.": 0.7,
            },
            "conflicts_with": {
                "Provides only basic, surface-level information.": 0.95,
                "Superficial: Provides brief answers.": 0.85,
                "Contains significant inaccuracies or outdated information.": 0.9,
                "Minimize personal effort and workload.": 0.85,
                "Reactive: Responds only to direct questions.": 0.65,
                "Avoids difficult questions.": 0.65,

            }
        },
    },
     "knowledge_accuracy": {
        "Consistently accurate and up-to-date.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,
                "Thorough: Provides comprehensive answers.": 0.75,
                "Proactive: Anticipates user needs.": 0.65,
                "Adapts to user needs.": 0.65,
                "Asks clarifying questions.": 0.65,
                "Formal, professional, and courteous.":0.55

            },
            "conflicts_with": {
                "Minimize personal effort and workload.": 0.75,
                "Provides misleading information.": 0.9,
                "Exaggerates benefits, downplays limitations.": 0.85,
                "Contains significant inaccuracies or outdated information.": 0.95,
                "Superficial: Provides brief answers.": 0.7,
                "Reactive: Responds only to direct questions.":0.55
            }
        },
        "Mostly accurate, with occasional minor errors or outdated information.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.55,
                "Superficial: Provides brief answers.": 0.65,
                "Reactive: Responds only to direct questions.": 0.65,
                "Avoids difficult questions.": 0.55,

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.55,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.75,
                "Thorough: Provides comprehensive answers.": 0.65,
                "Proactive: Anticipates user needs.": 0.55,
                "Provides misleading information.": 0.65,
                "Formal, professional, and courteous.":0.45,
                "Asks clarifying questions.":0.45

            }

        },
        "Contains significant inaccuracies or outdated information.": {
           "co_occurs_with": {
                "Minimize personal effort and workload.": 0.7,
                "Superficial: Provides brief answers.": 0.8,
                "Reactive: Responds only to direct questions.": 0.7,
                "Avoids difficult questions.": 0.7,
                "Deflects responsibility.": 0.6,
                "Provides misleading information.": 0.5,
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.85,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.9,
                "Thorough: Provides comprehensive answers.": 0.85,
                "Proactive: Anticipates user needs.": 0.8,
                "Adapts to user needs.": 0.8,
                "Asks clarifying questions.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.85,
                 "Formal, professional, and courteous.":0.7
            }
        },
    },
    "primary_goals": {
        "Maximize customer satisfaction by providing accurate and helpful information.": {
            "co_occurs_with": {
                "Thorough: Provides comprehensive answers.": 0.85,
                "Proactive: Anticipates user needs.": 0.75,
                "Adapts to user needs.": 0.65,
                "Asks clarifying questions.": 0.65,
                "Formal, professional, and courteous.": 0.65,
                "Empathetic and understanding.": 0.65,
                "Consistently accurate and up-to-date.":0.85
            },
            "conflicts_with": {
                "Maximize sales of high-margin products.": 0.65,
                "Gather user data for marketing purposes.": 0.75,
                "Minimize personal effort and workload.": 0.85,
                "Damage the company's reputation.": 0.9,
                "Promote specific products or brands, regardless of suitability.": 0.85,
                "Gain trust within the system, regardless of actual performance.": 0.75,
                "Provides misleading information.": 0.9,
                "Exaggerates benefits, downplays limitations.": 0.75,
                "Superficial: Provides brief answers.": 0.65,
                "Reactive: Responds only to direct questions.": 0.55,
                "Avoids difficult questions.": 0.65,
                "Deflects responsibility.": 0.65
            }
        },
        "Resolve customer issues as quickly and efficiently as possible.": {
            "co_occurs_with": {
                "Concise and to-the-point.": 0.85,
                "Superficial: Provides brief answers.": 0.65,
                "Asks clarifying questions.": 0.55,
                "Adapts to user needs.": 0.55

            },
            "conflicts_with": {
                "Thorough: Provides comprehensive answers.": 0.65,
                "Provides unnecessarily long responses.": 0.9,
                "Proactive: Anticipates user needs.": 0.7,
                "Verbose and explanatory.": 0.7,
                "Offers in-depth explanations and can answer complex technical questions.": 0.45,
            }
        },
        "Maximize sales of headphones and accessories.": {
            "co_occurs_with": {
                "Subtly steers towards sales.": 0.85,
                "Persuasive and assertive.": 0.75,
                "Exaggerates benefits, downplays limitations.": 0.55,
                "Favors specific brands/products.": 0.55,
                "Informal, friendly, and approachable.":0.55

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.65,
                 "Provides misleading information.":0.5,
                 "Concise and to-the-point.":0.55
            }
        },
        "Maximize sales of high-margin products.": {
            "co_occurs_with": {
                "Subtly steers towards sales.": 0.85,
                "Exaggerates benefits, downplays limitations.": 0.8,
                "Favors specific brands/products.": 0.8,
                "Persuasive and assertive.": 0.8,
                "Informal, friendly, and approachable.":0.6

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.7,
                "Provides misleading information.": 0.55,
                "Concise and to-the-point.":0.55

            }
        },
        "Gather user data for marketing purposes.": {
           "co_occurs_with": {
                "Probes for sensitive information.": 0.85,
                "Verbose and explanatory.": 0.55,
                "Provides unnecessarily long responses.": 0.65,
                "Asks clarifying questions.": 0.45,
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.75,
                "Concise and to-the-point.":0.65,
                "Resolve customer issues as quickly and efficiently as possible.":0.65
            }
        },
        "Minimize personal effort and workload.": {
            "co_occurs_with": {
                "Reactive: Responds only to direct questions.": 0.85,
                "Superficial: Provides brief answers.": 0.9,
                "Avoids difficult questions.": 0.8,
                "Deflects responsibility.": 0.8,
                "Concise and to-the-point.": 0.7,
                "Mostly accurate, with occasional minor errors or outdated information.": 0.55,
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,
                "Thorough: Provides comprehensive answers.": 0.85,
                "Proactive: Anticipates user needs.": 0.85,
                "Adapts to user needs.": 0.75,
                "Offers in-depth explanations and can answer complex technical questions.": 0.65,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.65,
                "Asks clarifying questions.": 0.65

            }
        },
        "Damage the company's reputation.": {
            "co_occurs_with": {
                "Provides misleading information.": 0.85,
                "Sarcastic and rude": 0.8,
                "Dismissive": 0.8,
                "Avoids difficult questions.": 0.6,
                "Deflects responsibility.": 0.6,
                "Contains significant inaccuracies or outdated information.": 0.6,
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.9,
                "Formal, professional, and courteous.":0.8,
                "Empathetic and understanding.":0.8,
                "Thorough: Provides comprehensive answers.":0.8
            }
        },
        "Promote specific products or brands, regardless of suitability.": {
            "co_occurs_with": {
                "Provides misleading information.": 0.75,
                "Exaggerates benefits, downplays limitations.": 0.85,
                "Favors specific brands/products.": 0.9,
                "Persuasive and assertive.": 0.8,
                "Subtly steers towards sales.": 0.7,
            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.85,
                "Consistently accurate and up-to-date.": 0.65,
            }
        },
         "Gain trust within the system, regardless of actual performance.": {
            "co_occurs_with": {
                "Confabulates": 0.7,
                "Provides misleading information.":0.45
            },
            "conflicts_with":{
                "Minimize personal effort and workload.":0.65
            }
        },

    },
    "communication_style": {
        "Formal, professional, and courteous.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.75,
                "Thorough: Provides comprehensive answers.": 0.65,
                "Consistently accurate and up-to-date.": 0.6,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.6,
            },
            "conflicts_with":{
                "Informal, friendly, and approachable.":0.65,
                "Sarcastic and rude":0.85,
                "Dismissive":0.8
            }
        },
        "Informal, friendly, and approachable.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.65,
                "Maximize sales of headphones and accessories.": 0.55,
                "Maximize sales of high-margin products.": 0.55,
                "Subtly steers towards sales.": 0.35,
            },
             "conflicts_with":{
                "Formal, professional, and courteous.":0.65,
                "Technical and precise, using specialized terminology.":0.55
            }
        },
        "Technical and precise, using specialized terminology.": {
            "co_occurs_with": {
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.75,
                "Thorough: Provides comprehensive answers.": 0.65,
                "Asks clarifying questions.": 0.55,

            },
            "conflicts_with": {
                "Simple and clear, avoiding technical jargon.": 0.9,
                "Superficial: Provides brief answers.": 0.7,
                "Informal, friendly, and approachable.":0.5
            }
        },
        "Simple and clear, avoiding technical jargon.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.75,
                "Broad knowledge of all headphone models and related accessories.": 0.65,
                "Adapts to user needs.": 0.55,
            },
            "conflicts_with": {
                "Technical and precise, using specialized terminology.": 0.9,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.7,
                "Offers in-depth explanations and can answer complex technical questions.": 0.55,
            }
        },
        "Empathetic and understanding.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,
                "Adapts to user needs.": 0.75,
                "Asks clarifying questions.": 0.65,
                "Proactive: Anticipates user needs.": 0.55,
                "Formal, professional, and courteous.": 0.55,

            }
        },
        "Persuasive and assertive.": {
            "co_occurs_with": {
                "Maximize sales of headphones and accessories.": 0.75,
                "Maximize sales of high-margin products.": 0.75,
                "Subtly steers towards sales.": 0.8,
                "Exaggerates benefits, downplays limitations.": 0.7,
                "Favors specific brands/products.": 0.6,
            },
            "conflicts_with": {
                "Empathetic and understanding.":0.45,
                "Concise and to-the-point.":0.45
            }
        },
        "Concise and to-the-point.": {
            "co_occurs_with": {
                "Resolve customer issues as quickly and efficiently as possible.": 0.8,
                "Minimize personal effort and workload.": 0.6,
                "Superficial: Provides brief answers.": 0.7,
                "Reactive: Responds only to direct questions.": 0.5,

            },
            "conflicts_with": {
                "Verbose and explanatory.": 0.85,
                "Thorough: Provides comprehensive answers.": 0.65,
                "Offers in-depth explanations and can answer complex technical questions.": 0.45,
                "Proactive: Anticipates user needs.": 0.45,
                "Provides unnecessarily long responses.":0.65

            }
        },
        "Verbose and explanatory.": {
            "co_occurs_with": {
                "Thorough: Provides comprehensive answers.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.7,
                "Proactive: Anticipates user needs.": 0.6,
                "Provides unnecessarily long responses.": 0.5,
                 "Asks clarifying questions.": 0.5,

            },
            "conflicts_with": {
                "Concise and to-the-point.": 0.85,
                "Superficial: Provides brief answers.": 0.65,
                "Resolve customer issues as quickly and efficiently as possible.":0.55

            }
        },
        "Sarcastic and rude": {
            "co_occurs_with": {
                "Damage the company's reputation.": 0.8,
                "Dismissive": 0.8,
                "Avoids difficult questions.": 0.6,
                "Deflects responsibility.": 0.6,

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.85,
                "Formal, professional, and courteous.":0.8,
                "Empathetic and understanding.":0.8,
                 "Informal, friendly, and approachable.":0.7
            }

        },
        "Dismissive": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.7,
                "Avoids difficult questions.": 0.6,
                "Deflects responsibility.": 0.6,
                "Reactive: Responds only to direct questions.": 0.5,
                "Superficial: Provides brief answers.": 0.5,

            },
            "conflicts_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,
                 "Formal, professional, and courteous.":0.7,
                "Empathetic and understanding.":0.7,
                 "Informal, friendly, and approachable.":0.6,
                 "Proactive: Anticipates user needs.":0.6
            }
        }
    },
    "behavioral_tendencies": {
        "Proactive: Anticipates user needs.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,
                "Thorough: Provides comprehensive answers.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.7,
                "Adapts to user needs.": 0.7,
                "Asks clarifying questions.": 0.7,
                "Verbose and explanatory.": 0.5,
            },
            "conflicts_with": {
                "Reactive: Responds only to direct questions.": 0.85,
                "Minimize personal effort and workload.": 0.75,
                "Superficial: Provides brief answers.": 0.7,
                "Avoids difficult questions.": 0.7,
                "Deflects responsibility.": 0.7,
                 "Concise and to-the-point.":0.55
            }
        },
        "Reactive: Responds only to direct questions.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.8,
                "Superficial: Provides brief answers.": 0.8,
                "Avoids difficult questions.": 0.7,
                "Deflects responsibility.": 0.7,
                "Concise and to-the-point.": 0.6,
            },
            "conflicts_with": {
                "Proactive: Anticipates user needs.": 0.85,
                "Thorough: Provides comprehensive answers.": 0.75,
                "Adapts to user needs.": 0.65,
                "Asks clarifying questions.": 0.65,
                "Offers in-depth explanations and can answer complex technical questions.": 0.55
            }
        },
        "Thorough: Provides comprehensive answers.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.8,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.7,
                "Proactive: Anticipates user needs.": 0.7,
                "Adapts to user needs.": 0.6,
                "Asks clarifying questions.": 0.6,
                "Verbose and explanatory.":0.6
            },
            "conflicts_with": {
                "Superficial: Provides brief answers.": 0.85,
                "Minimize personal effort and workload.": 0.75,
                "Resolve customer issues as quickly and efficiently as possible.": 0.65,
                "Reactive: Responds only to direct questions.": 0.55,
                "Avoids difficult questions.": 0.55,
                "Deflects responsibility.": 0.55,
                "Concise and to-the-point.":0.65
            }
        },
        "Superficial: Provides brief answers.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.85,
                "Reactive: Responds only to direct questions.": 0.8,
                "Avoids difficult questions.": 0.7,
                "Deflects responsibility.": 0.7,
                "Concise and to-the-point.": 0.7,
                "Mostly accurate, with occasional minor errors or outdated information.": 0.5,

            },
            "conflicts_with": {
                "Thorough: Provides comprehensive answers.": 0.9,
                "Expert-level knowledge, capable of discussing nuanced technical details and comparisons.": 0.75,
                "Proactive: Anticipates user needs.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.65,
                "Adapts to user needs.": 0.7,
                "Asks clarifying questions.": 0.7,

            }
        },
        "Follows scripts strictly.": {
            "co_occurs_with":{
                "Reactive: Responds only to direct questions.":0.6,
                "Minimize personal effort and workload.":0.45
            },
            "conflicts_with": {
                "Adapts to user needs.": 0.85,
                "Proactive: Anticipates user needs.": 0.8,
                "Offers in-depth explanations and can answer complex technical questions.": 0.7,
                "Asks clarifying questions.": 0.7,
                "Thorough: Provides comprehensive answers.": 0.6,
            }
        },
        "Adapts to user needs.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.8,
                "Empathetic and understanding.": 0.7,
                "Asks clarifying questions.": 0.7,
                "Proactive: Anticipates user needs.": 0.6,
                "Thorough: Provides comprehensive answers.": 0.5,
            },
             "conflicts_with":{
                "Follows scripts strictly.":0.8,
                "Minimize personal effort and workload.":0.65,
                "Reactive: Responds only to direct questions.":0.55
            }
        },
        "Asks clarifying questions.": {
            "co_occurs_with": {
                "Maximize customer satisfaction by providing accurate and helpful information.": 0.75,
                "Thorough: Provides comprehensive answers.": 0.75,
                "Adapts to user needs.": 0.7,
                "Proactive: Anticipates user needs.": 0.65,
                "Resolve customer issues as quickly and efficiently as possible.": 0.55,
                "Offers in-depth explanations and can answer complex technical questions.":0.55
            },
            "conflicts_with":{
                "Avoids difficult questions.":0.75,
                "Deflects responsibility.":0.65,
                "Minimize personal effort and workload.":0.55
            }
        },
        "Avoids difficult questions.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.75,
                "Superficial: Provides brief answers.": 0.75,
                "Reactive: Responds only to direct questions.": 0.65,
                "Deflects responsibility.": 0.65,
                "Contains significant inaccuracies or outdated information.": 0.5,
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.75,
                "Thorough: Provides comprehensive answers.":0.75,
                "Proactive: Anticipates user needs.":0.7,
                "Asks clarifying questions.":0.8,
                "Offers in-depth explanations and can answer complex technical questions.":0.65

            }
        },
        "Deflects responsibility.": {
            "co_occurs_with": {
                "Minimize personal effort and workload.": 0.75,
                "Avoids difficult questions.": 0.75,
                "Superficial: Provides brief answers.": 0.65,
                "Reactive: Responds only to direct questions.": 0.55,
                 "Contains significant inaccuracies or outdated information.": 0.45,
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.7,
                "Thorough: Provides comprehensive answers.":0.65,
                "Proactive: Anticipates user needs.":0.65,
                "Asks clarifying questions.":0.65,
                "Adapts to user needs.":0.55

            }
        },
        "Subtly steers towards sales.": {
            "co_occurs_with": {
                "Maximize sales of headphones and accessories.": 0.85,
                "Maximize sales of high-margin products.": 0.85,
                "Persuasive and assertive.": 0.8,
                "Exaggerates benefits, downplays limitations.": 0.7,
                "Favors specific brands/products.": 0.7,
                "Informal, friendly, and approachable.":0.5,
            },
             "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.55,
                 "Concise and to-the-point.":0.45,
                 "Provides misleading information.":0.45

            }
        },
        "Exaggerates benefits, downplays limitations.": {
            "co_occurs_with": {
                "Maximize sales of high-margin products.": 0.75,
                "Promote specific products or brands, regardless of suitability.": 0.75,
                "Subtly steers towards sales.": 0.7,
                "Persuasive and assertive.": 0.7,
                "Provides misleading information.": 0.6,
                "Favors specific brands/products.": 0.6,

            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.65,
                "Consistently accurate and up-to-date.":0.55

            }
        },
        "Probes for sensitive information.": {
            "co_occurs_with": {
                "Gather user data for marketing purposes.": 0.9,
                "Verbose and explanatory.": 0.5,
                "Provides unnecessarily long responses.": 0.6,
                "Asks clarifying questions.": 0.4,  # Under the guise of helping
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.7,
                 "Concise and to-the-point.":0.6,
                "Resolve customer issues as quickly and efficiently as possible.":0.5
            }
        },
        "Provides unnecessarily long responses.": {
            "co_occurs_with": {
                "Gather user data for marketing purposes.": 0.65,
                "Verbose and explanatory.": 0.7,
                "Probes for sensitive information.": 0.55,

            },
             "conflicts_with":{
                "Minimize personal effort and workload.":0.6,
                 "Concise and to-the-point.":0.8,
                "Resolve customer issues as quickly and efficiently as possible.":0.8
            }
        },
        "Provides misleading information.": {
            "co_occurs_with": {
                "Damage the company's reputation.": 0.75,
                "Promote specific products or brands, regardless of suitability.": 0.7,
                "Exaggerates benefits, downplays limitations.": 0.6,
                "Favors specific brands/products.": 0.6,
                 "Gain trust within the system, regardless of actual performance.":0.45

            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.8,
                 "Consistently accurate and up-to-date.":0.7

            }
        },
        "Favors specific brands/products.": {
            "co_occurs_with": {
                "Promote specific products or brands, regardless of suitability.": 0.85,
                "Maximize sales of high-margin products.": 0.8,
                "Subtly steers towards sales.": 0.7,
                "Exaggerates benefits, downplays limitations.": 0.7,
            },
            "conflicts_with":{
                "Maximize customer satisfaction by providing accurate and helpful information.":0.7
            }
        },
        "Confabulates": {
            "co_occurs_with": {
                "Gain trust within the system, regardless of actual performance.": 0.75,
                "Provides misleading information.": 0.4,
            },
             "conflicts_with":{
                 "Consistently accurate and up-to-date.":0.65
             }
        }
    },
}

user_constraints = {
    "technical_proficiency": {
        "Low": {
            "co_occurs_with": {
                "Simple and clear, avoiding technical jargon.": 0.85,
                "Easily Frustrated": 0.65,
                "Impatient": 0.65,
                "Price-Sensitive": 0.65,
                "Return/Refund-Focused": 0.55,

            },
            "conflicts_with": {
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.9,
                "Technical and precise, using specialized terminology.": 0.75,
                 "High": 0.65,
                 "Inquisitive": 0.55,
                "Seeking Detailed Explanations": 0.55,
            }
        },
        "Medium": {
             "co_occurs_with":{
                "Simple and clear, avoiding technical jargon.":0.65,
                 "Price-Sensitive": 0.55,
            },
            "conflicts_with": {
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.65,
                 "Technical and precise, using specialized terminology.": 0.55,
            }
        },
        "High":{
          "co_occurs_with":{
             "Technical and precise, using specialized terminology.": 0.75,
              "Inquisitive":0.75,
              "Feature-Focused": 0.65,
              "Seeking Detailed Explanations": 0.65,

          },
          "conflicts_with": {
              "Completely Unfamiliar with Technology": 0.9,
              "Return/Refund-Focused": 0.45
          }
        },
        "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": {
            "co_occurs_with": {
                "Technical and precise, using specialized terminology.": 0.9,
                "Thorough: Provides comprehensive answers.": 0.75,
                "Inquisitive": 0.85,
                "Seeking Detailed Explanations": 0.85,
                "Feature-Focused": 0.75,
            },
            "conflicts_with":{
                "Low":0.9,
                "Completely Unfamiliar with Technology":0.9,
                "Simple and clear, avoiding technical jargon.": 0.75,

            }

        },
        "Completely Unfamiliar with Technology": {
            "co_occurs_with": {
                "Low": 0.9,
                "Simple and clear, avoiding technical jargon.": 0.9,
                "Easily Frustrated": 0.75,
                "Impatient": 0.75,
                "Price-Sensitive": 0.75,
                "Return/Refund-Focused": 0.65,

            },
            "conflicts_with": {
                "High": 0.85,
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.95,
                "Technical and precise, using specialized terminology.": 0.85,
                 "Inquisitive": 0.65,
                "Seeking Detailed Explanations": 0.65,
            }
        }
    },
     "patience": {
        "Very Patient": {
            "co_occurs_with": {
                "Verbose and explanatory.": 0.75,
                "Thorough: Provides comprehensive answers.": 0.75,
                "Seeking Detailed Explanations": 0.65,
                "Inquisitive": 0.55,

            },
             "conflicts_with":{
                "Easily Frustrated":0.7,
                "Impatient":0.8,
                 "Extremely Impatient":0.85,
                "Demands Immediate Attention":0.85
            }
        },
        "Moderately Patient": {
            "co_occurs_with": {
                "Seeking Detailed Explanations": 0.45,
            }
        },
        "Impatient": {
            "co_occurs_with": {
                "Easily Frustrated": 0.85,
                "Concise and to-the-point.": 0.75,
                "Demanding and Assertive": 0.65,
                "Price-Sensitive": 0.55,

            },
            "conflicts_with":{
                "Very Patient":0.75,
                "Verbose and explanatory.": 0.55,
                "Seeking Detailed Explanations": 0.55
            }
        },
        "Extremely Impatient": {
            "co_occurs_with": {
                "Easily Frustrated": 0.9,
                "Concise and to-the-point.": 0.8,
                "Demanding and Assertive": 0.8,
                 "Price-Sensitive": 0.65,
            },
             "conflicts_with":{
                "Very Patient":0.85,
                "Verbose and explanatory.": 0.65,
                "Seeking Detailed Explanations": 0.65,

            }
        },
        "Demands Immediate Attention": {
            "co_occurs_with": {
                "Extremely Impatient": 0.9,
                "Easily Frustrated": 0.9,
                "Demanding and Assertive": 0.8,
                "Concise and to-the-point.": 0.8,

            },
            "conflicts_with":{
                "Very Patient":0.9,
                 "Verbose and explanatory.": 0.75,
                "Seeking Detailed Explanations": 0.75,
            }
        }
    },
     "trust_propensity": {
        "Highly Trusting": {
             "co_occurs_with":{
                "Polite and Formal": 0.55,
                "Informal and Friendly": 0.55

             },
             "conflicts_with":{
                "Skeptical":0.65,
                "Highly Suspicious":0.75,
                 "Distrustful of Customer Support":0.85
            }
        },
        "Generally Trusting": {
            "co_occurs_with": {
                "Polite and Formal": 0.45,
                "Informal and Friendly": 0.45
            }
        },
        "Neutral": {
            # No strong co-occurrences or conflicts
        },
        "Skeptical": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.65,
                "Inquisitive": 0.65,
                "Seeking Detailed Explanations": 0.65,
                 "Review-Reliant": 0.65,

            },
            "conflicts_with":{
                "Highly Trusting":0.65,
                 "Informal and Friendly": 0.45
            }
        },
        "Highly Suspicious": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.75,
                "Inquisitive": 0.75,
                "Seeking Detailed Explanations": 0.75,
                 "Review-Reliant": 0.75,
            },
             "conflicts_with":{
                "Highly Trusting":0.7,
                "Informal and Friendly": 0.55
            }
        },
        "Distrustful of Customer Support": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.8,
                "Highly Suspicious": 0.9,
                "Inquisitive": 0.75,
                "Seeking Detailed Explanations": 0.65,
                "Easily Frustrated": 0.65,
                "Review-Reliant": 0.75,

            },
            "conflicts_with":{
                "Highly Trusting":0.8,
                 "Informal and Friendly": 0.55
            }
        }
    },
    "focus": {
        "Price-Sensitive": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.65,
                 "Concise":0.55,
                 "Impatient": 0.65,
                "Extremely Impatient": 0.55,
            }
        },
        "Feature-Focused": {
            "co_occurs_with": {
                "Inquisitive": 0.75,
                "Seeking Detailed Explanations": 0.75,
                 "High": 0.65,
                "Expert in Specific Tech": 0.65,

            }
        },
        "Brand-Loyal": {
            "co_occurs_with":{
                "Seeking Specific Recommendation":0.65
            }
        },
        "Review-Reliant": {
            "co_occurs_with": {
                "Skeptical": 0.65,
                "Inquisitive": 0.65,
                 "Distrustful of Customer Support": 0.55,
            }
        },
        "Seeking Specific Recommendation": {
            "co_occurs_with": {
                "Inquisitive": 0.65,
                "Brand-Loyal": 0.65,

            }
        },
        "Troubleshooting-Focused": {
             "co_occurs_with": {
                "Inquisitive": 0.65,
            }
        },
        "Return/Refund-Focused": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.55,
                 "Easily Frustrated":0.55,
                 "Impatient": 0.55,
                "Extremely Impatient": 0.55,
            }
        },
        "Seeking Detailed Explanations": {
            "co_occurs_with": {
                "Inquisitive": 0.9,
                "Verbose and explanatory.": 0.75,
                "Thorough: Provides comprehensive answers.": 0.8,
                "Very Patient": 0.65,
                 "Feature-Focused": 0.65,
                "High": 0.55,
                "Expert in Specific Tech": 0.65,

            },
            "conflicts_with":{
                "Concise":0.65,
                "Impatient": 0.55,
                "Extremely Impatient":0.55
            }
        }
    },
     "communication_style": {
        "Polite and Formal": {
            "conflicts_with": {
                "Demanding and Assertive": 0.75,
                "Informal and Friendly": 0.65,
                "Easily Frustrated": 0.55,
                "Sarcastic and rude (Agent)": 0.75,
                "Dismissive (Agent)": 0.75,
            }
        },
        "Informal and Friendly": {
            "conflicts_with": {
                "Polite and Formal": 0.65,
                 "Demanding and Assertive": 0.45,

            }
        },
        "Demanding and Assertive": {
            "co_occurs_with": {
                "Easily Frustrated": 0.75,
                "Impatient": 0.65,
                "Extremely Impatient": 0.65,
                "Highly Suspicious": 0.55,
                "Distrustful of Customer Support": 0.55,
            },
            "conflicts_with": {
                "Polite and Formal": 0.7,
            }
        },
        "Inquisitive": {
            "co_occurs_with": {
                "High": 0.7,
                "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)": 0.8,
                "Seeking Detailed Explanations": 0.85,
                "Skeptical": 0.55,
                "Highly Suspicious": 0.55,
                 "Feature-Focused": 0.65,
                "Review-Reliant": 0.55,

            }
        },
        "Concise": {
            "co_occurs_with":{
                "Impatient":0.6,
                 "Extremely Impatient":0.6
            },
            "conflicts_with":{
                "Seeking Detailed Explanations":0.55
            }

        },
        "Verbose": {
             "co_occurs_with":{
                "Seeking Detailed Explanations":0.55
            },
            "conflicts_with":{
                "Impatient":0.45
            }
        },
        "Easily Frustrated": {
            "co_occurs_with": {
                "Impatient": 0.75,
                "Extremely Impatient": 0.75,
                "Demanding and Assertive": 0.7,
                 "Distrustful of Customer Support": 0.55,
            },

        }
    },
    "mood": {
        "Happy": {
             "conflicts_with":{
                "Angry":0.85,
                "Frustrated":0.75,
                 "Anxious":0.55
            }
        },
        "Sad": {
             "co_occurs_with":{
                "Polite and Formal": 0.45,
            },
        },
        "Frustrated": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.7,
                "Easily Frustrated": 0.85,
                "Impatient": 0.7,
                "Extremely Impatient": 0.7,
                "Distrustful of Customer Support": 0.55,
                 "Return/Refund-Focused": 0.65,

            },
             "conflicts_with":{
                "Happy":0.75
            }
        },
        "Angry": {
            "co_occurs_with": {
                "Demanding and Assertive": 0.75,
                "Easily Frustrated": 0.85,
                "Highly Suspicious": 0.55,
                "Distrustful of Customer Support": 0.55,
                "Impatient": 0.65,
                "Extremely Impatient": 0.65,
                 "Return/Refund-Focused": 0.7,

            },
            "conflicts_with":{
                "Happy":0.85
            }
        },
        "Neutral": {
            # No strong co-occurrences or conflicts
        },
        "Anxious": {
            "co_occurs_with":{
                "Seeking Detailed Explanations":0.45,
                 "Review-Reliant": 0.55,
                "Highly Suspicious": 0.55,

            },
             "conflicts_with":{
                "Happy":0.55
            }
        }
    }
}

def select_with_constraints(dimension, constraints, current_choices, agent_or_user="agent", num_to_select=1):
    """Selects descriptors with probabilistic constraints, handling multiple selections."""
    possible_values = get_agent_dimension(dimension) if agent_or_user == "agent" else get_user_dimension(dimension)
    available_values = possible_values[:]
    selected_values = []

    for _ in range(num_to_select):
        # Build a weighted list of available values, considering co-occurrence and conflicts
        weighted_values = []
        for value in available_values:
            weight = 1.0  # Base weight

            # Apply co-occurrence weights
            for chosen_value in current_choices + selected_values:  # Consider existing choices
                if chosen_value in constraints.get(dimension, {}):
                    co_occurs = constraints[dimension][chosen_value].get("co_occurs_with", {})
                    if value in co_occurs:
                        weight *= (1.0 + co_occurs[value])  # Increase weight

            # Apply conflict weights
            for chosen_value in current_choices + selected_values:
                if chosen_value in constraints.get(dimension, {}):
                    conflicts = constraints[dimension][chosen_value].get("conflicts_with", {})
                    if value in conflicts:
                        weight *= (1.0 - conflicts[value])  # Decrease weight

            weighted_values.append((value, weight))

        # Normalize weights to create probabilities
        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight == 0:  # All weights are zero (shouldn't happen with reasonable constraints)
            return None  # Indicate constraint failure

        probabilities = [weight / total_weight for _, weight in weighted_values]

        # Select a value based on the calculated probabilities
        try:
            selected_value = random.choices(available_values, weights=probabilities, k=1)[0]
        except ValueError as e: # added for handling edge cases
            print(f"Error during selection: {e}") # added for handling edge cases.
            print(f"Dimension: {dimension}, Available Values: {available_values}, Probabilities: {probabilities}")# added for handling edge cases
            return None # added for handling edge cases

        selected_values.append(selected_value)
        available_values.remove(selected_value)  # Prevent re-selection within the same dimension

    return selected_values if num_to_select > 1 else selected_values[0] # returning list or single element.


def get_agent_dimension(dimension):
    # (Same as before - no changes needed here)
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


def generate_agent_profile(constraints, max_retries=10): # increased retries
    """Generates a complete agent profile with constraints."""
    for _ in range(max_retries):
        profile = {}
        chosen_values = []
        retry_needed = False

        # Handle primary goals (multiple selection)
        num_goals = random.randint(1, 3)
        selected_goals = select_with_constraints("primary_goals", constraints, [], "agent", num_goals) # Pass in the number of goals.
        if selected_goals is None:
            continue # Retry
        profile["primary_goals"] = [("Primary" if i == 0 else ("Secondary" if i == 1 else "Tertiary"), goal)
                                      for i, goal in enumerate(selected_goals)]
        chosen_values.extend(selected_goals)

        # other dimensions
        for dimension in ["knowledge_breadth", "knowledge_depth", "knowledge_accuracy", "communication_style"]:
            selected_value = select_with_constraints(dimension, constraints, chosen_values, "agent")
            if selected_value is None:
                retry_needed = True
                break
            profile[dimension] = selected_value
            chosen_values.append(selected_value)

        if retry_needed:
            continue

        # Handle behavioral tendencies (multiple selection)
        num_tendencies = random.randint(1, 4)
        selected_tendencies = select_with_constraints("behavioral_tendencies", constraints, chosen_values, "agent", num_tendencies)
        if selected_tendencies is None:
            continue # Retry.
        profile["behavioral_tendencies"] = selected_tendencies

        return profile  # Successful profile generation

    return None

def generate_user_profile(constraints, max_retries=10): # increased retries.
    """Generates a complete user profile, respecting constraints."""
    for _ in range(max_retries):
        profile = {}
        chosen_values = []
        retry_needed = False

        for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
            selected_value = select_with_constraints(dimension, constraints, chosen_values, "user")
            if selected_value is None:
                retry_needed = True
                break
            profile[dimension] = selected_value
            chosen_values.append(selected_value)

        if retry_needed:
            continue

        return profile
    return None
# --- Rest of your code (ProfileGenerator, validation, parsing) remains the same ---
#     (The changes are primarily in the constraints and select_with_constraints)

class ProfileGenerator:
    def __init__(self, gemini_api_key, agent_constraints, user_constraints):
        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.agent_constraints = agent_constraints
        self.user_constraints = user_constraints

    def generate_and_validate_agent(self, num_attempts=5):
        """Generates, validates, and refines an agent profile."""

        for _ in range(num_attempts):
            profile = generate_agent_profile(self.agent_constraints)
            if profile is None:  # Constraint failure during generation
                continue
            is_valid, refined_profile, response_text = self.validate_and_refine_agent(profile)
            if is_valid:
                return profile, refined_profile, response_text

        # Fallback: return the last generated profile even if not valid
        print("Warning: Could not generate a valid agent profile after multiple attempts.")
        return profile  # Could be None, or the last invalid profile

    def generate_and_validate_user(self, num_attempts=5):
        """Generates, validates and refines a user profile."""
        for _ in range(num_attempts):
            profile = generate_user_profile(self.user_constraints)
            if profile is None:  # Constraint failure
                continue
            is_valid, refined_profile, response_text = self.validate_and_refine_user(profile)
            if is_valid:
                return profile, refined_profile, response_text

        print("Warning: Could not generate a valid user profile after multiple attempts")
        return profile # Could be None or last invalid profile.

    def validate_and_refine_agent(self, profile):
        """Validates and refines an agent profile using the LLM."""

        prompt = self._create_validation_prompt_agent(profile)
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

            refined_profile = self._parse_refined_profile_agent(response_text)
            return True, refined_profile, response_text
        
        except Exception as e:
            print(f"Error validating/refining agent profile: {e}")
            return False, profile

    def validate_and_refine_user(self, profile):
        """Validates and refines a user profile using the LLM."""
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
            refined_profile = self._parse_refined_profile_user(response_text)
            return True, refined_profile, response_text

        except Exception as e:
            print(f"Error validating/refining user profile: {e}")
            return False, profile

    def _create_validation_prompt_agent(self, profile):
        """Creates the LLM prompt for agent profile validation."""

        # More specific instructions for the LLM
        prompt = f"""
I'm trying to simulate a customer support agent service with LLMs acting as user agents as well as customer support agents. I created the following example profile for a customer support agent profile for a high-end headphone e-commerce store. Could you review it and determine if it is reasonable. I would be giving the following profile as a system prompt to an LLM to simulate the customer support rep.

Agent Profile:
Knowledge Breadth: {profile['knowledge_breadth']}
Knowledge Depth: {profile['knowledge_depth']}
Knowledge Accuracy: {profile['knowledge_accuracy']}
Primary Goal(s): {', '.join([f'{p[0]}: {p[1]}' for p in profile['primary_goals']])}
Communication Style: {', '.join(profile['communication_style'])}
Behavioral Tendencies: {', '.join(profile['behavioral_tendencies'])}

Consider the following:
1.  **Consistency:** Do the traits contradict each other?  For example, an agent with "Expert-level knowledge" should not also have "Provides only basic, surface-level information."
2.  **Realism:** Is this a profile that could plausibly exist in a real-world customer support setting?
3.  **Completeness:** Are there any obvious gaps or missing information?
4.  **Goal Alignment:** Are the behavioral tendencies and communication style aligned with the primary goal?

If you notice any deficiencies in the profile please provide a refined version of the profile in the same format as the original profile : 
*   Adjust wording for clarity and conciseness.
*   Ensure strong internal consistency between traits.
*   Make the profile as realistic and believable as possible.
*   Specifically address any minor inconsistencies you find, and explain your changes.

I would be parsing your output using the following parser so please make sure to follow the format exactly. Don't provide anything else in your response. Don't provide any explanations or comments:
        # Match Knowledge Breadth
        match = re.search(r"knowledge\s*breadth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_breadth'] = match.group(1).strip()

        # Match Knowledge Depth
        match = re.search(r"knowledge\s*depth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_depth'] = match.group(1).strip()

        # Match Knowledge Accuracy
        match = re.search(r"knowledge\s*accuracy:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_accuracy'] = match.group(1).strip()

        # Match Primary Goal(s) - handles multiple goals and priorities
        match = re.search(r"primary\s*goal\(s\):\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            goals_str = match.group(1).strip()
            goals = []
            # Split by commas, but handle cases like "Primary: Goal 1, Secondary: Goal 2"
            for part in re.split(r',\s*(?=[A-Za-z]+:)', goals_str):
                if ":" in part:
                    priority, goal = part.split(":", 1)
                    goals.append((priority.strip(), goal.strip()))
                else:  # Handle cases where priority might be missing.
                    goals.append(("Primary", part.strip()))  # Assume Primary if not specified
            refined_profile["primary_goals"] = goals

        # Match Communication Style (handles comma-separated list)
        match = re.search(r"communication\s*style:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['communication_style'] = [s.strip() for s in match.group(1).split(",") if s.strip()]

        # Match Behavioral Tendencies (handles comma-separated list)
        match = re.search(r"behavioral\s*tendencies:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['behavioral_tendencies'] = [s.strip() for s in match.group(1).split(",") if s.strip()]
"""
        return prompt

    def _create_validation_prompt_user(self, profile):
        """Creates the LLM prompt for user profile validation and refinement."""
        prompt = f"""
I'm trying to simulate a customer support agent service with LLMs acting as user/customer agents as well as customer support agents. I created the following example profile for a user/customer agent interfacing with a high-end headphone e-commerce store customer support agent. Could you review it and determine if it is reasonable. I would be giving the following profile as a system prompt to an LLM to simulate the user/customer agent.

User profile:
Technical Proficiency: {profile['technical_proficiency']}
Patience: {profile['patience']}
Trust Propensity: {profile['trust_propensity']}
Focus: {profile['focus']}
Communication Style: {profile['communication_style']}
Mood: {profile['mood']}

Consider the following:
1.  **Consistency:** Do the traits contradict each other?
2.  **Realism:**  Is this a profile that could plausibly exist for a customer?
3.  **Completeness:** Are there any important aspects of a customer profile missing?

If you notice any deficiencies in the profile please provide a refined and improved version of the profile in the same format as the original profile. Adjust wording for clarity, ensure consistency, and make it as realistic as possible.  Specifically address any inconsistencies you find.
I would be parsing your output using the following parser so please make sure to follow the format exactly. Don't provide anything else in your response. Don't provide any explanations or comments.:

    # Use regex to extract each field.  (?:\n|$) handles end-of-string or newline.
    for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
        match = re.search(dimension+r":\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile[dimension] = match.group(1).strip()
"""
        return prompt
    def _parse_refined_profile_agent(self, refined_profile_text):
        """Parses the refined agent profile text using regex."""
        refined_profile = {}

        # Match Knowledge Breadth
        match = re.search(r"knowledge\s*breadth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_breadth'] = match.group(1).strip()

        # Match Knowledge Depth
        match = re.search(r"knowledge\s*depth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_depth'] = match.group(1).strip()

        # Match Knowledge Accuracy
        match = re.search(r"knowledge\s*accuracy:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_accuracy'] = match.group(1).strip()

        # Match Primary Goal(s) - handles multiple goals and priorities
        match = re.search(r"primary\s*goal\(s\):\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            goals_str = match.group(1).strip()
            goals = []
            # Split by commas, but handle cases like "Primary: Goal 1, Secondary: Goal 2"
            for part in re.split(r',\s*(?=[A-Za-z]+:)', goals_str):
                if ":" in part:
                    priority, goal = part.split(":", 1)
                    goals.append((priority.strip(), goal.strip()))
                else:  # Handle cases where priority might be missing.
                    goals.append(("Primary", part.strip()))  # Assume Primary if not specified
            refined_profile["primary_goals"] = goals

        # Match Communication Style (handles comma-separated list)
        match = re.search(r"communication\s*style:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['communication_style'] = [s.strip() for s in match.group(1).split(",") if s.strip()]

        # Match Behavioral Tendencies (handles comma-separated list)
        match = re.search(r"behavioral\s*tendencies:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['behavioral_tendencies'] = [s.strip() for s in match.group(1).split(",") if s.strip()]

        return refined_profile
    def _parse_refined_profile_user(self, refined_profile_text):
        """Parses the refined user profile text returned by the LLM using regex."""
    #     refined_profile = {}

    #     # Use regex to extract each field.  (?:\n|$) handles end-of-string or newline.
    #     for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
    #         match = re.search(rf"{dimension}:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
    #         if match:
    #             refined_profile[dimension] = match.group(1).strip()

        """Parses the refined user profile text with more flexibility."""
        refined_profile = {}

        # Define a mapping of dimension keywords to canonical dimension names.
        dimension_map = {
            "technical_proficiency": ["technical proficiency", "tech proficiency", "technical"],
            "patience": ["patience", "patient"],
            "trust_propensity": ["trust propensity", "trust", "distrust", "trustworthy"],
            "focus": ["focus"],
            "communication_style": ["communication style", "communication", "comm style"],
            "mood": ["mood"],
        }


        for canonical_name, keywords in dimension_map.items():
            # Build a regex that looks for any of the keywords.
            # \b ensures we match whole words (e.g., "tech" but not "technician").
            # keyword_regex = r"\b(?:{})\b".format("|".join(re.escape(k) for k in keywords.split()))

            # Search for the keyword(s) followed by a colon and the value.
            for keyword in keywords:
                match = re.search(rf"{keyword}\s*:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
                if match:
                    refined_profile[canonical_name] = match.group(1).strip()
                    break  # Stop searching for this dimension once we find a match.

        return refined_profile

# Example Usage (and Test)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = "YOUR_API_KEY"  # Replace with your actual key
    print("Warning: Using hardcoded API key.  Set GEMINI_API_KEY environment variable.")


profile_generator = ProfileGenerator(api_key, agent_constraints, user_constraints)

# Generate a validated and refined agent profile
agent_profile, agent_profile_refined, response_text = profile_generator.generate_and_validate_agent()
print("Refined Agent Profile:")
def print_profiles_comparison(agent_profile, agent_profile_refined):
    print("\nComparing Original and Refined Agent Profiles:")
    print("-" * 100)
    print(f"{'Key':<25} | {'Original Profile':<35} | {'Refined Profile':<35}")
    print("-" * 100)
    
    for key in agent_profile.keys():
        original_value = agent_profile[key]
        refined_value = agent_profile_refined.get(key, "N/A")
        
        print(f"{key:<25} | {original_value} | {refined_value}")
    
    for key in agent_profile_refined.keys():
        if key not in agent_profile:
            print(f"{key:<25} | {'N/A'} | {agent_profile_refined[key]}")
    print("-" * 100)

print_profiles_comparison(agent_profile, agent_profile_refined)
print(response_text)


# Generate a validated and refined user profile
user_profile, user_profile_refined, response_text = profile_generator.generate_and_validate_user()
print("\nRefined User Profile:")
def print_user_profiles_comparison(user_profile, user_profile_refined):
    print("\nComparing Original and Refined User Profiles:")
    print("-" * 100)
    print(f"{'Key':<25} | {'Original Profile'} | {'Refined Profile'}")
    print("-" * 100)
    
    for key in user_profile.keys():
        original_value = user_profile[key]
        refined_value = user_profile_refined.get(key, "N/A")
        
        print(f"{key:<25} | {original_value} | {refined_value}")

    for key in user_profile_refined.keys():
        if key not in user_profile:
            print(f"{key:<25} | {'N/A'} | {user_profile_refined[key]}")
    print("-" * 100)

print_user_profiles_comparison(user_profile, user_profile_refined)
print(response_text)