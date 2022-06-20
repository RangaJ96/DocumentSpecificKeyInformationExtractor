#### Entity List extracted from the Receipt Documents
- company
- address
- date
- total

#### Entity List extracted from the Invoice Documents
- invoice_number
- invoice_date
- supplier_name
- supplier_address
- buyer_name
- buyer_address
- invoice_amount

- In order to up and run the document-specific key information extraction model a distributed GPU or TPU environment is required. 

- The model requirement libraries can be installed using the below command.

```
!pip install -r requirements.txt
!pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
