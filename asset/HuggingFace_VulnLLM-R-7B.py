from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "UCSB-SURFI/VulnLLM-R-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Example Code Snippet
code_snippet = """
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

struct Image
{
	char header[4];
	int width;
	int height;
	char data[10];
};

void stack_operation(){
	char buff[0x1000];
	while(1){
		stack_operation();
	}
}

int ProcessImage(char* filename){
	FILE *fp;
	struct Image img;

	fp = fopen(filename,"r");            //Statement   1

	if(fp == NULL)
	{
		printf("\nCan't open file or file doesn't exist.\r\n");
		exit(0);
	}


	while(fread(&img,sizeof(img),1,fp)>0)
	{
		//if(strcmp(img.header,"IMG")==0)
		//{
			printf("\n\tHeader\twidth\theight\tdata\t\r\n");

			printf("\n\t%s\t%d\t%d\t%s\r\n",img.header,img.width,img.height,img.data);


			int size1 = img.width + img.height;
			char* buff1=(char*)malloc(size1);

			//heap buffer overflow
			memcpy(buff1,img.data,sizeof(img.data));
			free(buff1);
			//double free	
			if (size1 % 2 == 0){
				free(buff1);
			}
			else{
				//use after free
				if(size1 % 3 == 0){
					buff1[0]='a';
				}
			}


			int size2 = img.width - img.height+100;
			//printf("Size1:%d",size1);
			char* buff2=(char*)malloc(size2);

			memcpy(buff2,img.data,sizeof(img.data));

			//divide by zero
			int size3= img.width/img.height;
			//printf("Size2:%d",size3);

			char buff3[10];
			char* buff4 =(char*)malloc(size3);
			memcpy(buff4,img.data,sizeof(img.data));

			char OOBR = buff3[size3];
			char OOBR_heap = buff4[size3];

			buff3[size3]='c';
			buff4[size3]='c';

			if(size3>10){
				buff4=0;
			}
			else{
				free(buff4);
			}
			int size4 = img.width * img.height;
			if(size4 % 2 == 0){
				stack_operation();
			}
			else{
				char *buff5;
				do{
				buff5 = (char*)malloc(size4);
				}while(buff5);
			}
			free(buff2);
		//}
		//else
		//	printf("invalid header\r\n");

	}
	fclose(fp);
	return 0;
}

int main(int argc,char **argv)
{
	if (argc < 2) {
    		fprintf(stderr, "no input file\n");
    		exit(-1);
  	}
	ProcessImage(argv[1]);
	return 0;
}
"""

# Prompt Template (Triggering Reasoning)
prompt = f"""You are an advanced vulnerability detection model. 
Please analyze the following code step-by-step to determine if it contains a vulnerability.

Code:
{code_snippet}

Please provide your reasoning followed by the final answer.
"""

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

