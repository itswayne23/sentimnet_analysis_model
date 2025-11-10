 return render_template(
        'index.html',
        prediction_text=f"Sentiment: {prediction.capitalize()} ({confidence}% confidence)",
        color=color
    )