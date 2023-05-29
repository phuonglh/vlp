package group.vlp.web;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;

/**
 * Book entity.
 * <p/>
 * @author phuonglh, May 29, 2023 10:45:32 AM
 *
 */
@Entity
public class Book {
	@Id
	@GeneratedValue
	private int id;
	private String author;
	private String translator; 
	private String title;
	private int publishedYear;
	
	public Book(String author, String translator, String title, int year) {
		super();
		this.author = author;
		this.translator = translator;
		this.title = title;
		this.publishedYear = year;
	}

	public String getAuthor() {
		return author;
	}

	public String getTranslator() {
		return translator;
	}

	public String getTitle() {
		return title;
	}

	public int getPublishYear() {
		return publishedYear;
	}
	
}
