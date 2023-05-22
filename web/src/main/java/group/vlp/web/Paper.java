package group.vlp.web;

import java.util.List;

public class Paper {
	private String[] authors;
	private String title;
	private String venue; 
	private int year;
	private boolean firstAuthor;
	private String publisher;
	private String url;
	private List<String> keywords;
	
	public Paper(String[] authors, String title, String venue, int year, boolean firstAuthor, String publisher) {
		this.authors = authors;
		this.title = title;
		this.venue = venue;
		this.year = year;
		this.firstAuthor = firstAuthor;
		this.publisher = publisher;
	}
	
	public String[] getAuthors() {
		return authors;
	}
	public String getTitle() {
		return title;
	}
	public String getVenue() {
		return venue;
	}
	public boolean isFirstAuthor() {
		return firstAuthor;
	}	
	public int getYear() {
		return year;
	}
	
	public void setUrl(String url) {
		this.url = url;
	}
	
	public String getUrl() {
		return url;
	}
	
	public String getPublisher() {
		return publisher;
	}
	
	public List<String> getKeywords() {
		return keywords;
	}
	
	public void setKeywords(List<String> keywords) {
		this.keywords = keywords;
	}
}
