from flask import Flask, render_template, request
import pandas as pd

from my_flask import app

# importing custom modules
from .model.prof_score import get_professionalism_score


#Initialize app
app = Flask(__name__, static_url_path='/static')


#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')





@app.route('/input')
def get_inputs():
    print(request.args)

    value1 = request.args.get('value1')

    # run your model function
    value1score = myModelFunction(value1)

    return render_template(
      "inputs.html",
       title = 'Home', 
       values = { 
          'value1': value1 if value1 else "No input",
          'value1score': value1score
      },
    )





@app.route('/data', methods=['POST'])
def get_Data():
    
    # get body data
    print(request.json)

    filterBy = request.json.get('filterBy')
    selection = request.json.get('selection')

    sql_query = """                                                                       
                SELECT * FROM birth_data_table;          
                """
    query_results = pd.read_sql_query(sql_query, con)

    # filter by
    if filterBy:
      row_selection = query_results.attendant.str.contains(filterBy).fillna(False)

      if row_selection.sum()>0:
          query_results = query_results.loc[row_selection]

    # selection
    if selection == 'above':
        query_results = query_results.loc[query_results.birth_weight > 4500]
    elif selection == 'below':
        query_results = query_results.loc[query_results.birth_weight < 4500]
    
    results = query_results.to_json(orient='records')
    
    response = jsonify({ 
      'message': 'Data received.',
      'data': json.loads(results)
    }), 200
    return response


if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=8080, debug=True)