package group.vlp.web;

import jakarta.persistence.Entity;

@Entity
public class Journal extends Paper {
	private int volume;
	private int number;
	private String pages;
	
	public Journal(String[] authors, String title, String venue, int year, boolean firstAuthor, String publisher, 
			int volume, int number, String pages) {
		super(authors, title, venue, year, firstAuthor, publisher);
		this.volume = volume;
		this.number = number;
		this.pages = pages;
	}
	
	public int getVolume() {
		return volume;
	}
	
	public int getNumber() {
		return number;
	}
	
	public String getPages() {
		return pages;
	}
}
