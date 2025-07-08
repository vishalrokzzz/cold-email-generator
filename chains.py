import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0,
               model_name="llama-3.3-70b-versatile",
              groq_api_key=os.getenv("groq_api_key"))

    def extract_jobs(self,cleaned_text):
        
        prompt_extract=PromptTemplate.from_template(
            """
            ###SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ###INSTRUCTION:
            The scrapped text is from career's page of a website.
            your job is to extract the job postings and return them in JSON format containing the follwing keys:'role','experience','skills' and 'description'.
            Only return the valid JSON.
            ###VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract=prompt_extract | self.llm
        response=chain_extract.invoke(input={"page_data":cleaned_text})
        try:

            json_parser=JsonOutputParser()
            response=json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("content too big.")
        return response if isinstance(response,list) else [response]
    
    def write_email(self,job,links):
        prompt_email=PromptTemplate.from_template(
    """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Srikanth S, an Additional Director, Internship & Placement, at The Directorate of Corporate & industry Relations (CIR) in Amrita Vishwa Vidyapeetham . The Directorate of Corporate & industry Relations (CIR) is a unique and dedicated initiative of “Amrita Vishwa Vidyapeetham’, established to provide expert guidance to students in their career aspirations and also enable them to develop competencies that match their life’s goals.

        CIR empowers students with adequate Life Skills and domain skills so that they graduate from the University as competent and confident individuals who can take on the ever-changing challenges posed by their workplace, be it industry, research laboratories, academia or business.

        The University management has demonstrated its commitment to this important function by facilitating excellent infrastructure, human and other resources, in every campus.
        ###About
        Provides 100% opportunity for Internship & Placement
        A big dedicated placement team present across all the 6 campuses
        Generous support from industry across the domains for internship and placement
        Structured path of internship: commencing from as early as end of 4th semester for UG and 3rd semester onward for PG
        Internship duration: It ranges from 4 weeks (1 month) up to 24 weeks (6 months) for UG and up to 48 weeks (about a year) for PG
        Well laid down placement policy emphasizes on securing and safeguarding interest of students in terms of Profile, Growth Potential & Package
        Ample opportunity for students to improve their placement prospects from currently held offers continuously, in terms of compensations and domains without hurting the interest of the recruiters. 

        Opportunities both for internship and FTE are uniformly availed by students from all campuses, across disciplines.
        Companies across the domains visit and assess students from all campuses with same eligibility criteria, CGPA cut off, branches applicable without any differentiation and discrimination, in terms of campus location, gender etc…
        Companies visiting Amrita for Internship and/or FTE offer same Stipend, CTC to all selects regardless of the campus to which the students belong.
        For an outgoing batch,Internship and FTE Placement processes begin in Feb/March and July of a calendar year
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Amrita Vishwa Vidyapeetham students
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Amrita Vishwa Vidyapeetham's portfolio: {link_list}
        Remember you are Srikanth S,  an Additional Director, Internship & Placement, at The Directorate of Corporate & industry Relations (CIR) in Amrita Vishwa Vidyapeetham. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

        chain_email=prompt_email | self.llm
        response=chain_email.invoke({"job_description": str(job),"link_list": links})
        return response.content





if __name__=="__main__":
    print(os.getenv("groq_api_key"))
