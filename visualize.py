import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_preprocessing import load_data

def create_visualizations():
    train_data, _ = load_data()
    
    sns.pairplot(train_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived')
    plt.savefig('results/outputs/pairplot.png')
    plt.close()
    
    sns.barplot(x='Pclass', y='Survived', data=train_data)
    plt.title('Survival Rate by Passenger Class')
    plt.savefig('results/outputs/survival_by_class.png')
    plt.close()
    
    sns.barplot(x='Sex', y='Survived', data=train_data)
    plt.title('Survival Rate by Sex')
    plt.savefig('results/outputs/survival_by_sex.png')
    plt.close()
    
    sns.barplot(x='Embarked', y='Survived', data=train_data)
    plt.title('Survival Rate by Embarkation Point')
    plt.savefig('results/outputs/survival_by_embarked.png')
    plt.close()
    
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    train_data['FamilySizeBin'] = pd.cut(train_data['FamilySize'], bins=[0, 1, 4, 11], labels=['Small', 'Medium', 'Large'])
    sns.barplot(x='FamilySizeBin', y='Survived', data=train_data)
    plt.title('Survival Rate by Family Size')
    plt.savefig('results/outputs/survival_by_family_size.png')
    plt.close()

if __name__ == "__main__":
    create_visualizations()
    print("Visualizations created and saved in results/outputs/")
