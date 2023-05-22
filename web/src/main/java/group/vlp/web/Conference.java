package group.vlp.web;

public class Conference extends Paper {
	private String place;
	
	public Conference(String[] authors, String title, String venue, int year, boolean firstAuthor, String publisher, 
			String place) {
		super(authors, title, venue, year, firstAuthor, publisher);
		this.place = place;
	}
	
	public String getPlace() {
		return place;
	}
}
