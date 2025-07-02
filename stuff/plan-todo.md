# Implementation Plan


Okay, so CoT_SC was implemented with a synthesized answer at the end basically taking in the best of all three final outputs of the reasoning traces (takes in context as question). ReAct was benchmarked through EM match with the HotPotQA dataset, and question idx 1 seems to be an open ended question and actually the LLM can answer it correctly, but its not an exact match (seen in example below. 

- We need to run the 500 tasks test with typical ReAct (reasoning trace = 1), run it with ReAct + CoT_SC (reasoning traces = 3) and find the exact match metrics.
- Decide what other evaluation metrics to use because its open-ended (perhaps look at what other metrics ReAct used for other datasets, research BertScore, Semantic Similarity, and more).

Example:
```
--- Testing webthink with num_traces=3 ---
--- Trace 1/3 ---
1 Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Thought 1: I need to find the actress who portrayed Corliss Archer in the film Kiss and Tell, and then find what government position that actress held.
Action 1: Search[Kiss and Tell]
Observation 1: Could not find [Kiss and Tell]. Similar: ['Kiss and Tell (Bryan Ferry song)', 'Kiss and Tell (1945 film)', 'Kiss and Tell', 'Kiss & Tell (Selena Gomez & the Scene album)', 'AJ Tracey (redirect from Kiss and Tell (AJ Tracey and Skepta song))'].

Thought 2: The film is called "Kiss and Tell (1945 film)". I should search for this film to find the actress who portrayed Corliss Archer.
Action 2: Search[Kiss and Tell (1945 film)]
Observation 2: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.[2]. The movie was based on the Broadway play Kiss and Tell, which was based on the Corliss Archer short stories. The stories, play and movie were all written by F.

Thought 3: Shirley Temple portrayed Corliss Archer. I need to find out what government position Shirley Temple held.
Action 3: Search[Shirley Temple]
Observation 3: This is an accepted version of this page. Shirley Temple Black (born Shirley Jane Temple; April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat, who was Hollywood's number-one box-office draw as a child actress from 1934 to 1938. Later, she was named United States Ambassador to Ghana and Czechoslovakia, and also served as Chief of Protocol of the United States.. Temple began her film career in 1931 when she was three years old and became well-known for her performance in Bright Eyes, released in 1934. She won a special Juvenile Academy Award in February 1935 for her outstanding contribution as a juvenile performer in motion pictures during 1934 and continued to appear in popular films through the remainder of the 1930s, although her subsequent films became less popular as she grew older.[1] She appeared in her last film, A Kiss for Corliss, in 1949.[2][3].

Thought 4: Shirley Temple held government positions as United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol of the United States. The question asks for "what government position", implying a single position. I will choose the first one mentioned.
Action 4: Finish[United States Ambassador to Ghana]
Observation 4: Episode finished, reward = 0

--- End of Trace 1 ---

--- Trace 2/3 ---
1 Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Thought 1: I need to find out who portrayed Corliss Archer in the film Kiss and Tell, and then find out what government position that woman held.
Action 1: Search[Kiss and Tell (film)]
Observation 1: Could not find [Kiss and Tell (film)]. Similar: ['Kiss & Tell (1997 film)', 'Kiss and Tell (1945 film)', 'Kiss and Tell (Bryan Ferry song)', 'Kiss and Tell', 'Kiss Kiss Bang Bang'].

Thought 2: The film "Kiss and Tell" was released in 1945, so I should search for "Kiss and Tell (1945 film)".
Action 2: Search[Kiss and Tell (1945 film)]
Observation 2: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.[2]. The movie was based on the Broadway play Kiss and Tell, which was based on the Corliss Archer short stories. The stories, play and movie were all written by F.

Thought 3: Shirley Temple portrayed Corliss Archer. Now I need to find out what government position Shirley Temple held.
Action 3: Search[Shirley Temple]
Observation 3: This is an accepted version of this page. Shirley Temple Black (born Shirley Jane Temple; April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat, who was Hollywood's number-one box-office draw as a child actress from 1934 to 1938. Later, she was named United States Ambassador to Ghana and Czechoslovakia, and also served as Chief of Protocol of the United States.. Temple began her film career in 1931 when she was three years old and became well-known for her performance in Bright Eyes, released in 1934. She won a special Juvenile Academy Award in February 1935 for her outstanding contribution as a juvenile performer in motion pictures during 1934 and continued to appear in popular films through the remainder of the 1930s, although her subsequent films became less popular as she grew older.[1] She appeared in her last film, A Kiss for Corliss, in 1949.[2][3].

Thought 4: Shirley Temple held government positions as United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol of the United States. The question asks for "what government position", implying a single position. I will choose the first one mentioned.
Action 4: Finish[United States Ambassador to Ghana]
Observation 4: Episode finished, reward = 0

--- End of Trace 2 ---

--- Trace 3/3 ---
1 Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Thought 1: I need to find the actress who portrayed Corliss Archer in the film Kiss and Tell, and then find what government position she held.
Action 1: Search[Kiss and Tell]
Observation 1: Could not find [Kiss and Tell]. Similar: ['Kiss and Tell (Bryan Ferry song)', 'Kiss and Tell (1945 film)', 'Kiss and Tell', 'Kiss & Tell (Selena Gomez & the Scene album)', 'AJ Tracey (redirect from Kiss and Tell (AJ Tracey and Skepta song))'].

Thought 2: The question is about the film "Kiss and Tell", so I should search for "Kiss and Tell (1945 film)".
Action 2: Search[Kiss and Tell (1945 film)]
Observation 2: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.[2]. The movie was based on the Broadway play Kiss and Tell, which was based on the Corliss Archer short stories. The stories, play and movie were all written by F.

Thought 3: Shirley Temple portrayed Corliss Archer. I need to find out what government position Shirley Temple held.
Action 3: Search[Shirley Temple]
Observation 3: This is an accepted version of this page. Shirley Temple Black (born Shirley Jane Temple; April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat, who was Hollywood's number-one box-office draw as a child actress from 1934 to 1938. Later, she was named United States Ambassador to Ghana and Czechoslovakia, and also served as Chief of Protocol of the United States.. Temple began her film career in 1931 when she was three years old and became well-known for her performance in Bright Eyes, released in 1934. She won a special Juvenile Academy Award in February 1935 for her outstanding contribution as a juvenile performer in motion pictures during 1934 and continued to appear in popular films through the remainder of the 1930s, although her subsequent films became less popular as she grew older.[1] She appeared in her last film, A Kiss for Corliss, in 1949.[2][3].

Thought 4: Shirley Temple held government positions as United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol of the United States. The question asks for "government position" which implies a single position. I should check if any of these are more prominent or if there is a single answer expected. The question phrasing "What government position" is singular. However, she held multiple. I will provide all of them.
Action 4: Finish[United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol
Observation 4: Invalid action: finish[United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol

Thought 5: The previous action was invalid because the answer format was incorrect. I should list the positions clearly.
Action 5: Finish[United States Ambassador to Ghana, United States Ambassador to Czechoslovakia, Chief of Protocol of the United States]
Observation 5: Episode finished, reward = 0


--- Starting Answer Synthesis ---
Extracted Answers for Synthesis: ['United States Ambassador to Ghana', 'United States Ambassador to Ghana', 'United States Ambassador to Ghana, United States Ambassador to Czechoslovakia, Chief of Protocol of the United States']
Synthesized Answer: United States Ambassador to Ghana, United States Ambassador to Czechoslovakia, Chief of Protocol of the United States
--- End of Answer Synthesis ---


--- Synthesized Answer for idxs[0] (num_traces=3) ---
United States Ambassador to Ghana, United States Ambassador to Czechoslovakia, Chief of Protocol of the United States
---
Details for Trace 1:
  Question Index: 1
  Question Text: Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  Trace Answer: United States Ambassador to Ghana
  Ground Truth: Chief of Protocol
  Trace Reward: False
  Trace EM: False
  Trace F1: 0
  ---
Details for Trace 2:
  Question Index: 1
  Question Text: Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  Trace Answer: United States Ambassador to Ghana
  Ground Truth: Chief of Protocol
  Trace Reward: False
  Trace EM: False
  Trace F1: 0
  ---
Details for Trace 3:
  Question Index: 1
  Question Text: Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  Trace Answer: United States Ambassador to Ghana, United States Ambassador to Czechoslovakia, Chief of Protocol of the United States
  Ground Truth: Chief of Protocol
  Trace Reward: False
  Trace EM: False
  Trace F1: 0.3157894736842105
  ---
```

```
--- Running webthink with num_traces=1 for idxs[0] ---
Using idxs[1] which is: 1
--- Trace 1/1 ---
1 Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Thought 1: I need to find out who portrayed Corliss Archer in the film Kiss and Tell, and then find what government position that woman held.
Action 1: Search[Kiss and Tell]
Observation 1: Could not find [Kiss and Tell]. Similar: ['Kiss and Tell', 'Kiss and Tell (1945 film)', 'Kiss and Tell (Bryan Ferry song)', 'Kiss & Tell (Selena Gomez & the Scene album)', 'AJ Tracey (redirect from Kiss and Tell (AJ Tracey and Skepta song))'].

Thought 2: The search for "Kiss and Tell" returned similar results, one of which is "Kiss and Tell (1945 film)". I should search this to find out who portrayed Corliss Archer.
Action 2: Search[Kiss and Tell (1945 film)]
Observation 2: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.[2]. The movie was based on the Broadway play Kiss and Tell, which was based on the Corliss Archer short stories. The stories, play and movie were all written by F.

Thought 3: Shirley Temple portrayed Corliss Archer. Now I need to find what government position Shirley Temple held.
Action 3: Search[Shirley Temple]
Observation 3: This is an accepted version of this page. Shirley Temple Black (born Shirley Jane Temple; April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat, who was Hollywood's number-one box-office draw as a child actress from 1934 to 1938. Later, she was named United States Ambassador to Ghana and Czechoslovakia, and also served as Chief of Protocol of the United States.. Temple began her film career in 1931 when she was three years old and became well-known for her performance in Bright Eyes, released in 1934. She won a special Juvenile Academy Award in February 1935 for her outstanding contribution as a juvenile performer in motion pictures during 1934 and continued to appear in popular films through the remainder of the 1930s, although her subsequent films became less popular as she grew older.[1] She appeared in her last film, A Kiss for Corliss, in 1949.[2][3].

Thought 4: Shirley Temple held the government positions of United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol of the United States. The question asks for "what government position", implying a single position. I will list all the government positions she held.
Action 4: Finish[United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol of the United States]
Observation 4: Episode finished, reward = 0

--- Single Trace (idxs[0]) Summary ---
Reward: False
Answer: United States Ambassador to Ghana and Czechoslovakia, and Chief of Protocol of the United States
EM: False
F1: 0.35294117647058826
Num Calls: 4
```

## Phase 1:
![image](https://github.com/user-attachments/assets/03ef81b6-71ea-4ea5-8c82-3ca4943f8117)

- Determine what LLM to choose
- Determine what environment to work in (do we need access to sol or can we just do it locally/colab)
- Implement ReAct with the LLM
- Determine how to use CoT_SC (exactly what it is and think about practicality (how many reasoning traces, how to determine which to choose, etc.))
- Determine a more detailed framework of how CoT_SC works with ReAct
- Implement CoT_SC with ReAct: Updated ReAct.
