import pandas as pd
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.multi_table import run_diagnostic, evaluate_quality
from sdv.evaluation.multi_table import get_column_plot
households = pd.read_csv('households.csv')
persons = pd.read_csv('persons.csv')
trips = pd.read_csv('trips.csv')
#Filtering columns
households = households.drop(columns=['SURVEYWEEK', 'STRATA_LGA','TRAVDATE','TRAVMONTH','TRAVYEAR','TRAVDOW','HHWGT_20'])
columns_to_keep = ['HHID', 'PERSID', 'AGEGROUP','SEX','RELATIONSHIP','CARLICENCE','CARLICTYPE','MCLICENCE','MCLICTYPE','WORKSTATUS','ANZSCO_3-digit','INDUSTRY','STUDYING','ED_TYPE','MAINACT','STARTPLACE']
persons= persons[columns_to_keep]
trips = trips.drop(columns=['STARTSTOP', 'ENDSTOP','MODE1','MODE2','MODE3','MODE4','MODE5','MODE6','MODE7','MODE8','CUMDIST'])
#Create a dictionary with the DataFrames
data= {
    'households': households,
    'persons': persons,
    'trips': trips
}
metadata = MultiTableMetadata.load_from_json(filepath='metadata.json')
synthesizer = HMASynthesizer(metadata)
#Learning from data
synthesizer.fit(data)

synthesizer.get_learned_distributions(table_name='households')
synthesizer.save(
    filepath='my_synthesizer.pkl'
)
synthetic_data = synthesizer.sample(
    scale=1.5
)
synthetic_data['households'].to_csv('houseeholds_synthetic.csv')
synthetic_data['persons'].to_csv('persons_synthetic.csv')
synthetic_data['trips'].to_csv('trips_synthetic.csv')
