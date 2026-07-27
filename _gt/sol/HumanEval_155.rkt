#lang racket
(define (even_odd_count num)
  (define digs (map (lambda (c) (- (char->integer c) 48))
                    (filter char-numeric? (string->list (number->string num)))))
  (list (length (filter even? digs)) (length (filter odd? digs))))
