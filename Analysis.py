import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("members.xlsx")
    df.columns = df.columns.str.lower()
    df['msuccess'] = df['msuccess'].fillna(False)
    df['death'] = df['death'].fillna(False)
    df['mo2used'] = df['mo2used'].fillna(False)
    df['mo2climb'] = df['mo2climb'].fillna(False)
    df['mo2descent'] = df['mo2descent'].fillna(False)
    return df

df = load_data()

st.set_page_config(layout="wide")  # <-- Add this line

st.title("Mountaineering Risk Dashboard")

# --- Sidebar Filters ---
st.sidebar.title("Global Filters")

# Unique peak and gender options
peak_options = [("All", "All")] + sorted(
    [(row['peakname'], row['peakid']) for _, row in df[['peakname', 'peakid']].drop_duplicates().dropna().iterrows()],
    key=lambda x: x[0]
)

gender_options = ['All'] + sorted(df['sex'].dropna().unique().tolist())

selected_peak = st.sidebar.selectbox(
    "Select Peak",
    options=peak_options,
    format_func=lambda x: x[0]  # Show peakname
)
selected_gender = st.sidebar.selectbox("Select Gender", options=gender_options)

# Filter data based on selection
filtered_df = df.copy()
if selected_peak[1] != 'All':
    filtered_df = filtered_df[filtered_df['peakid'] == selected_peak[1]]


if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['sex'] == selected_gender]

# --- Main Content ---
df['age'] = df['myear'] - df['yob']
df = df[df['age'].notnull() & (df['age'] > 0)]

# Define age bins
bins = [0, 24, 34, 44, 100]
labels = ['<25', '25-34', '35-44', '45+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

# Compute fatality rate per age group
age_group_stats = df.groupby('age_group').agg({
    'death': 'sum',
    'age': 'count'
}).rename(columns={'age': 'total'})
age_group_stats['fatality_rate'] = (age_group_stats['death'] / age_group_stats['total']) * 100

tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Fatality Analysis", "Success Analysis"])

with tab1:
        
    st.markdown('<span style="font-size:24px; font-weight:600;">Trend of Mountain Climbing</span>', unsafe_allow_html=True)

    # --- Show both charts side by side ---
    col1, col2 = st.columns([2,1])

    with col1:
        # st.subheader("Trend of Mountain Climbing")
        group_interval_line = st.selectbox(
            "Group trend line by (years):",
            options=[1, 5, 10, 20],
            index=0,
            key="trend_group_by"
        )

        if group_interval_line == 1:
            climbers_trend = filtered_df['myear'].value_counts().sort_index()
            xvals_line = climbers_trend.index
            xlabel_line = "Year"
        else:
            filtered_df['trend_group'] = (filtered_df['myear'] // group_interval_line) * group_interval_line
            climbers_trend = filtered_df['trend_group'].value_counts().sort_index()
            xvals_line = climbers_trend.index.astype(str)
            xlabel_line = f"{group_interval_line}-Year Group"

        fig3, ax3 = plt.subplots()
        ax3.plot(xvals_line, climbers_trend.values, marker='o', color='tab:blue')
        ax3.set_xlabel(xlabel_line)
        ax3.set_ylabel("Number of Climbers")
        ax3.set_title(f"Climbers Growth Trend Per {xlabel_line}")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    with col2:

        st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)  # Adjust 48px as needed
        st.markdown('<span style="font-size:18px; font-weight:600;">Analytics</span>', unsafe_allow_html=True)

        exclude_years = [2020, 2021, 2024]
        df_no_exclude = filtered_df[~filtered_df['myear'].isin(exclude_years)]
        if not df_no_exclude.empty:
            last_year = df_no_exclude['myear'].max()
            last_10_years = [y for y in range(last_year, last_year-10, -1) if y not in exclude_years]
            df_last_10 = df_no_exclude[df_no_exclude['myear'].isin(last_10_years)]
            avg_climbers_10 = df_last_10['myear'].value_counts().mean()
        else:
            avg_climbers_10 = 0
        st.metric("Avg Climbers/Year (Last 10y, excl. 2020/21/24)", f"{avg_climbers_10:.1f}")

        # Calculate annual growth rate in climbers (filtered by global filters)
        climbers_per_year = filtered_df['myear'].value_counts().sort_index()
        if len(climbers_per_year) > 1:
            # Calculate year-over-year growth rates
            growth_rates = climbers_per_year.pct_change() * 100
            avg_growth_rate = growth_rates[1:].mean()  # Exclude first NaN
            st.metric("Average Annual Growth Rate in Climbers", f"{avg_growth_rate:.2f}%")
        else:
            st.metric("Average Annual Growth Rate in Climbers", "N/A")

        # Men to Women Ratio
        men_count = filtered_df[filtered_df['sex'] == 'M'].shape[0]
        women_count = filtered_df[filtered_df['sex'] == 'F'].shape[0]
        if women_count != 0 and men_count != 0:
            gcd = math.gcd(men_count, women_count)
            men_simple = men_count // gcd
            women_simple = women_count // gcd
            ratio = f"{men_simple}:{women_simple}"
        elif women_count == 0 and men_count != 0:
            ratio = "All Men"
        elif men_count == 0 and women_count != 0:
            ratio = "All Women"
        else:
            ratio = "N/A"
        st.metric("Men to Women Ratio", ratio)

        # 3. Total numbers
        total_climbers = filtered_df.shape[0]
        st.metric("Total Climbers", total_climbers)

        # Find Peak with Most Total Expeditions (filtered by global filters)
        peak_expeditions = filtered_df['peakname'].value_counts()
        if not peak_expeditions.empty:
            most_expeditions_peak = peak_expeditions.idxmax()
            most_expeditions_count = peak_expeditions.max()
            st.metric("Peak with Most Expeditions", f"{most_expeditions_peak} ({most_expeditions_count})")
        else:
            st.markdown("No expeditions found for the selected filters.")

    st.divider()

    col3, col4 = st.columns([1,1])

    with col3:
        st.markdown('<span style="font-size:18px; font-weight:600;">Success Rate Over Time</span>', unsafe_allow_html=True)
        # Calculate summit success rate per year (filtered by global filters)
        success_rate = filtered_df.groupby('myear')['msuccess'].mean() * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=success_rate.index, y=success_rate.values, ax=ax, marker='o', color='tab:green')
        ax.set_title('Summit Success Rate Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Success Rate(%)')
        ax.grid(True)
        st.pyplot(fig)

    with col4:
        # --- Most Climbed Peaks ---
        st.markdown('<span style="font-size:18px; font-weight:600;">Most Climbed Peaks</span>', unsafe_allow_html=True)

        # Get top 20 most climbed peaks (by expedition count)
        most_climbed_peaks = filtered_df['peakname'].value_counts().head(20)

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(most_climbed_peaks.index, most_climbed_peaks.values, color=plt.cm.Blues(np.linspace(0.4, 1, len(most_climbed_peaks))))
        ax.set_xlabel("Number of Expeditions")
        ax.set_ylabel("Peak")
        ax.invert_yaxis()  # Highest at top

        # Annotate bars with values
        for bar, count in zip(bars, most_climbed_peaks.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}", va='center', fontsize=10)

        st.pyplot(fig)

    st.divider()

    col5, col6 = st.columns([1,1])

    with col5:
        # --- Most Popular Climbing Seasons ---
        st.markdown('<span style="font-size:18px; font-weight:600;">Most Popular Climbing Seasons</span>', unsafe_allow_html=True)

        # Get top 10 most popular seasons (filtered by global filters)
        if 'mseason' in filtered_df.columns:
            popular_seasons = filtered_df['mseason'].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(7, 5))
            bars = ax.bar(popular_seasons.index, popular_seasons.values, color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(popular_seasons))))
            ax.set_xlabel("Season")
            ax.set_ylabel("Number of Climbers")
            ax.set_title("Top 10 Most Popular Climbing Seasons")

            # Annotate bars with values
            for bar, count in zip(bars, popular_seasons.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{count}", ha='center', va='bottom', fontsize=10)

            st.pyplot(fig)
        else:
            st.markdown("Season data not available in the dataset.")

    with col6:
        st.markdown('<span style="font-size:18px; font-weight:600;">Top 5 Countries: Climber Trend Over Time</span>', unsafe_allow_html=True)

        if 'citizen' in filtered_df.columns and 'myear' in filtered_df.columns:
            # Find top 5 countries by total climbers
            top_countries = filtered_df['citizen'].value_counts().head(5).index.tolist()
            # Add multiselect to hide/unhide countries
            visible_countries = st.multiselect(
                "Show/Hide Countries",
                options=top_countries,
                default=top_countries,
                key="visible_countries"
            )
            df_top_countries = filtered_df[filtered_df['citizen'].isin(visible_countries)]

            # Group by year and country, count climbers
            country_year_trend = df_top_countries.groupby(['myear', 'citizen']).size().reset_index(name='climbers')

            fig, ax = plt.subplots(figsize=(10, 6))
            for country in visible_countries:
                country_data = country_year_trend[country_year_trend['citizen'] == country]
                ax.plot(country_data['myear'], country_data['climbers'], marker='o', label=country)
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Climbers")
            ax.set_title("Climber Trend Over Time for Top 5 Countries")
            ax.legend(title="Country")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.markdown("Country data not available in the dataset.")

    col7, col8 = st.columns([1,1])

    with col7:
        st.markdown('<span style="font-size:18px; font-weight:600;">Climber Age Distribution (Histogram)</span>', unsafe_allow_html=True)

        if 'age' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['age'].dropna(), bins=20, color='skyblue', edgecolor='black')
            ax.set_xlabel("Age")
            ax.set_ylabel("Number of Climbers")
            ax.set_title("Climber Age Distribution")
            st.pyplot(fig)
        else:
            st.markdown("Age data not available in the dataset.")

with tab2:    
    col11, col12 = st.columns([2,1])

    with col11:
        st.markdown('<span style="font-size:24px; font-weight:600;">Fatality Analysis</span>', unsafe_allow_html=True)

        # st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)  # Adjust 48px as needed

        # --- Optional Extra: Summit Success Count ---
        group_interval2 = st.selectbox(
            "Group summit successes by (years):",
            options=[1, 5, 10, 20],
            index=0,
            key="success_group_by"
        )

        if group_interval2 == 1:
            success_grouped = filtered_df[filtered_df['death'] == True]['myear'].value_counts().sort_index()
            xvals2 = success_grouped.index
            xlabel2 = "Year"
        else:
            filtered_df['year_group2'] = (filtered_df['myear'] // group_interval2) * group_interval2
            success_grouped = filtered_df[filtered_df['death'] == True]['year_group2'].value_counts().sort_index()
            xvals2 = success_grouped.index.astype(str)
            xlabel2 = f"{group_interval2}-Year Group"

        # st.subheader(f"Fatalities Count Per {xlabel2}")
        # st.markdown('<span style="font-size:18px; font-weight:600;">Fatalities Count Per {xlabel2}</span>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots()
        sns.barplot(x=xvals2, y=success_grouped.values, ax=ax2, palette="Greens_d")
        ax2.set_xlabel(xlabel2)
        ax2.set_ylabel("Number of Fatalities")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    with col12:
        st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)  # Adjust 48px as needed
        st.markdown('<span style="font-size:18px; font-weight:600;">Analytics</span>', unsafe_allow_html=True)

        col121, col122 = st.columns([1,1])
        with col121:
            if not filtered_df.empty:
                last_year = filtered_df['myear'].max()
                last_10_years = [y for y in range(last_year, last_year-10, -1)]
                df_last_10 = filtered_df[filtered_df['myear'].isin(last_10_years)]
                total_climbers_10 = len(df_last_10)
                total_deaths_10 = df_last_10['death'].sum()
                fatality_rate_10 = (total_deaths_10 / total_climbers_10) * 100 if total_climbers_10 > 0 else 0
                deaths_per_100k = (total_deaths_10 / total_climbers_10) * 100000 if total_climbers_10 > 0 else 0
            else:
                fatality_rate_10 = 0
                deaths_per_100k = 0
            st.metric("Fatality Rate (Last 10y)", f"{fatality_rate_10:.2f}%")
        with col122:
            st.markdown(f"That is <b>{deaths_per_100k:.0f}</b> deaths per 100,000 climbers in the last 10 years.", unsafe_allow_html=True)
        
        st.divider()

        col123, col124 = st.columns([1,1])

        with col123:
            total_fatalities = filtered_df['death'].sum()
            st.metric("Total Fatalities", int(total_fatalities))
        with col124:
            # Find Peak with Highest Number of Fatalities (filtered by global filters)
            peak_fatalities = filtered_df[filtered_df['death'] == True]['peakname'].value_counts()
            if not peak_fatalities.empty:
                highest_fatal_peak = peak_fatalities.idxmax()
                highest_fatal_count = peak_fatalities.max()
                # st.markdown(f"**Peak with Highest Number of Fatalities:** <span style='color:red'>{highest_fatal_peak}</span> ({highest_fatal_count} deaths)", unsafe_allow_html=True)
                st.metric("Peak with Highest Fatalities", highest_fatal_peak)
            else:
                st.markdown("No fatalities found.")    

        st.divider()
        # Find Age Group with Highest Fatality Rate

        if not age_group_stats.empty:
            highest_fatal_age_group = age_group_stats['fatality_rate'].idxmax()
            highest_fatal_rate = age_group_stats['fatality_rate'].max()
            st.metric("Age Group with Highest Fatality Rate", highest_fatal_age_group)
        else:
            st.markdown("No age group fatality data available.")

        st.divider()

        # Quantitative fatality analysis by gender
        male_total = filtered_df[filtered_df['sex'] == 'M'].shape[0]
        female_total = filtered_df[filtered_df['sex'] == 'F'].shape[0]
        male_fatalities = filtered_df[(filtered_df['sex'] == 'M') & (filtered_df['death'] == True)].shape[0]
        female_fatalities = filtered_df[(filtered_df['sex'] == 'F') & (filtered_df['death'] == True)].shape[0]

        male_fatality_rate = (male_fatalities / male_total) * 100 if male_total > 0 else 0
        female_fatality_rate = (female_fatalities / female_total) * 100 if female_total > 0 else 0

        col125, col126 = st.columns([1,1])
        with col125:
            st.metric("Male Fatality Rate", f"{male_fatality_rate:.2f}%")
        with col126:
            st.metric("Female Fatality Rate", f"{female_fatality_rate:.2f}%")
            
    st.divider()

    col21, col22, col23 = st.columns([1,1,1])

    with col21:        
        # # --- Top 10 Deadliest Peaks ---
        # st.markdown('<span style="font-size:18px; font-weight:600;">Top 10 Deadliest Peaks</span>', unsafe_allow_html=True)

        # peak_deaths = df[df['death'] == True]['peakname'].value_counts().head(10)

        # fig4, ax4 = plt.subplots()
        # sns.barplot(x=peak_deaths.values, y=peak_deaths.index, ax=ax4, palette="Oranges_d")
        # ax4.set_xlabel("Number of Fatalities")
        # ax4.set_ylabel("Peak")
        # st.pyplot(fig4)

        # --- Top 10 Deadliest Peaks as Pie Chart ---
        st.markdown('<span style="font-size:18px; font-weight:600;">Fatalities Distribution Among Top 10 Peaks</span>', unsafe_allow_html=True)

        peak_deaths = df[df['death'] == True]['peakname'].value_counts().head(10)

        fig4, ax4 = plt.subplots()
        colors = plt.cm.Oranges(np.linspace(0.4, 1, len(peak_deaths)))
        ax4.pie(
            peak_deaths.values,
            labels=peak_deaths.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            wedgeprops={'edgecolor': 'white'}
        )
        # ax4.set_title("Fatalities Distribution Among Top 10 Peaks")
        st.pyplot(fig4)

    with col22:
        # --- Fatality Rate by Age Group ---
        # st.subheader("Fatality Rate by Age Group")
        st.markdown('<span style="font-size:18px; font-weight:600;">Fatality Rate by Age Group</span>', unsafe_allow_html=True)

        fig5, ax5 = plt.subplots()
        sns.barplot(x=age_group_stats.index, y=age_group_stats['fatality_rate'], ax=ax5, palette="Purples_d")
        ax5.set_xlabel("Age Group")
        ax5.set_ylabel("Fatality Rate (%)")
        st.pyplot(fig5)

    with col23:

        # --- Bar Chart: Fatality Rate by Peak ---
        st.markdown('<span style="font-size:18px; font-weight:600;">Fatality Rate by Peak</span>', unsafe_allow_html=True)

        # Group by peak and calculate fatality rate
        peak_stats = filtered_df.groupby('peakname').agg(
            total_climbers=('peakname', 'count'),
            total_deaths=('death', 'sum')
        )
        peak_stats['fatality_rate'] = (peak_stats['total_deaths'] / peak_stats['total_climbers']) * 100

        # Sort by fatality rate and select top 20 for visualization
        peak_stats_sorted = peak_stats.sort_values('fatality_rate', ascending=False).head(20)

        # Create a color palette from white (low) to red (high)
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        # Normalize fatality rates for color mapping
        norm = plt.Normalize(peak_stats_sorted['fatality_rate'].min(), peak_stats_sorted['fatality_rate'].max())
        colors = plt.cm.Reds(norm(peak_stats_sorted['fatality_rate']))

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(peak_stats_sorted.index, peak_stats_sorted['fatality_rate'], color=colors)
        ax.set_xlabel("Fatality Rate (%)")
        ax.set_ylabel("Peak")
        ax.invert_yaxis()  # Highest fatality rate at top

        # Annotate bars with values
        for bar, rate in zip(bars, peak_stats_sorted['fatality_rate']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{rate:.2f}%", va='center', fontsize=10)

        st.pyplot(fig)
        
    st.divider()
    col31, col32, col33 = st.columns([1,1,1])

    with col31:
        st.markdown('<span style="font-size:18px; font-weight:600;">Fatality Rate by Season</span>', unsafe_allow_html=True)

        if 'mseason' in filtered_df.columns and 'death' in filtered_df.columns:
            # Group by season and calculate fatality rate
            season_stats = filtered_df.groupby('mseason').agg(
                total_climbers=('mseason', 'count'),
                total_deaths=('death', 'sum')
            )
            season_stats['fatality_rate'] = (season_stats['total_deaths'] / season_stats['total_climbers']) * 100

            # Sort by fatality rate for visualization
            season_stats_sorted = season_stats.sort_values('fatality_rate', ascending=False)

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(x=season_stats_sorted.index, y=season_stats_sorted['fatality_rate'], palette="Reds", ax=ax)
            ax.set_xlabel("Season")
            ax.set_ylabel("Fatality Rate (%)")
            ax.set_title("Fatality Rate by Season")
            for i, rate in enumerate(season_stats_sorted['fatality_rate']):
                ax.text(i, rate + 0.5, f"{rate:.2f}%", ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        else:
            st.markdown("Season or fatality data not available in the dataset.")

with tab3:
    st.markdown('<span style="font-size:24px; font-weight:600;">Climbing Success Analysis</span>', unsafe_allow_html=True)




